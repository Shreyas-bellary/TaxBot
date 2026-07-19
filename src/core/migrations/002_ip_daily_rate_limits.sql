-- =============================================================================
-- IP daily rate-limit counter table
--
-- Stores one row per (client_ip, UTC calendar day). All state transitions
-- (consume, check, refund) are performed by the functions below, so callers
-- never need to write raw SQL against this table.
--
-- Cleanup is handled by a pg_cron job scheduled in the same migration.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS private;

-- -----------------------------------------------------------------------------
-- Core table
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS private.ip_daily_rate_limits (
    client_ip    INET        NOT NULL,
    usage_date   DATE        NOT NULL,
    answer_count INTEGER     NOT NULL DEFAULT 0
                             CHECK (answer_count >= 0),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (client_ip, usage_date)
);

-- Allows the cleanup DELETE and the nightly cron query to find old rows
-- without scanning the whole table.
CREATE INDEX IF NOT EXISTS ip_daily_rate_limits_usage_date_idx
    ON private.ip_daily_rate_limits (usage_date);

-- Keep updated_at current on every write.
CREATE OR REPLACE FUNCTION private.set_ip_rl_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS ip_daily_rate_limits_set_updated_at
    ON private.ip_daily_rate_limits;
CREATE TRIGGER ip_daily_rate_limits_set_updated_at
    BEFORE UPDATE ON private.ip_daily_rate_limits
    FOR EACH ROW EXECUTE FUNCTION private.set_ip_rl_updated_at();

-- -----------------------------------------------------------------------------
-- consume(p_client_ip, p_limit)
--
-- Tries to reserve one answer slot for today.
-- Returns a single row:
--   allowed       BOOLEAN  -- false when already at the limit before this call
--   answer_count  INTEGER  -- count AFTER the increment (only meaningful when allowed)
--   limit_val     INTEGER  -- the configured limit echoed back
--   reset_at      TIMESTAMPTZ  -- UTC midnight beginning the next calendar day
--
-- Race-condition safety: the row is locked with SELECT FOR UPDATE before
-- the counter is read and written, so concurrent calls from different
-- Cloud Run instances are serialised per (ip, day) without a full table lock.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION private.ip_rl_consume(
    p_client_ip TEXT,
    p_limit     INTEGER
)
RETURNS TABLE (
    allowed      BOOLEAN,
    answer_count INTEGER,
    limit_val    INTEGER,
    reset_at     TIMESTAMPTZ
)
LANGUAGE plpgsql AS $$
DECLARE
    v_today       DATE        := (now() AT TIME ZONE 'utc')::date;
    v_reset_at    TIMESTAMPTZ := (v_today + INTERVAL '1 day') AT TIME ZONE 'utc';
    v_count       INTEGER;
BEGIN
    -- Upsert a row for today so we always have something to lock.
    INSERT INTO private.ip_daily_rate_limits (client_ip, usage_date, answer_count)
    VALUES (p_client_ip::INET, v_today, 0)
    ON CONFLICT (client_ip, usage_date) DO NOTHING;

    -- Lock the row for the duration of this transaction so concurrent
    -- callers queue rather than read a stale count.
    SELECT rl.answer_count
    INTO   v_count
    FROM   private.ip_daily_rate_limits rl
    WHERE  rl.client_ip  = p_client_ip::INET
      AND  rl.usage_date = v_today
    FOR UPDATE;

    IF v_count >= p_limit THEN
        -- Already exhausted — do not increment.
        RETURN QUERY SELECT FALSE, v_count, p_limit, v_reset_at;
        RETURN;
    END IF;

    -- Increment and return the new count.
    UPDATE private.ip_daily_rate_limits rl
    SET    answer_count = rl.answer_count + 1
    WHERE  rl.client_ip  = p_client_ip::INET
      AND  rl.usage_date = v_today
    RETURNING rl.answer_count INTO v_count;

    RETURN QUERY SELECT TRUE, v_count, p_limit, v_reset_at;
END;
$$;

-- -----------------------------------------------------------------------------
-- check(p_client_ip, p_limit)
--
-- Read-only snapshot of the current quota state (no consume).
-- Returns the same columns as consume for a uniform caller interface.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION private.ip_rl_check(
    p_client_ip TEXT,
    p_limit     INTEGER
)
RETURNS TABLE (
    allowed      BOOLEAN,
    answer_count INTEGER,
    limit_val    INTEGER,
    reset_at     TIMESTAMPTZ
)
LANGUAGE plpgsql AS $$
DECLARE
    v_today    DATE        := (now() AT TIME ZONE 'utc')::date;
    v_reset_at TIMESTAMPTZ := (v_today + INTERVAL '1 day') AT TIME ZONE 'utc';
    v_count    INTEGER;
BEGIN
    SELECT rl.answer_count
    INTO   v_count
    FROM   private.ip_daily_rate_limits rl
    WHERE  rl.client_ip  = p_client_ip::INET
      AND  rl.usage_date = v_today;

    -- No row yet means zero consumed slots today.
    v_count := COALESCE(v_count, 0);

    RETURN QUERY SELECT v_count < p_limit, v_count, p_limit, v_reset_at;
END;
$$;

-- -----------------------------------------------------------------------------
-- refund(p_client_ip)
--
-- Decrements today's counter by one, clamping at zero.  Called when a
-- request fails before producing an answer (injection, retrieval, citation,
-- or security errors).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION private.ip_rl_refund(p_client_ip TEXT)
RETURNS VOID
LANGUAGE plpgsql AS $$
DECLARE
    v_today DATE := (now() AT TIME ZONE 'utc')::date;
BEGIN
    UPDATE private.ip_daily_rate_limits
    SET    answer_count = GREATEST(answer_count - 1, 0)
    WHERE  client_ip   = p_client_ip::INET
      AND  usage_date  = v_today
      AND  answer_count > 0;
    -- If no row exists or count is already 0 the UPDATE is a no-op, which is fine.
END;
$$;

-- -----------------------------------------------------------------------------
-- cleanup(p_retention_days)
--
-- Deletes rows older than p_retention_days UTC days (default 1 = yesterday
-- and older).  Called by the pg_cron job below and exposed so the app can
-- also trigger manual purges via a CLI script.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION private.ip_rl_cleanup(
    p_retention_days INTEGER DEFAULT 1
)
RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE
    v_deleted INTEGER;
BEGIN
    DELETE FROM private.ip_daily_rate_limits
    WHERE usage_date < (now() AT TIME ZONE 'utc')::date - (p_retention_days - 1);

    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$;

-- =============================================================================
-- pg_cron daily cleanup job (idempotent via cron.schedule upsert semantics)
--
-- Supabase enables pg_cron on the `postgres` database by default.  The call
-- to cron.schedule uses the job-name as the primary key, so re-running this
-- migration updates the schedule rather than inserting a duplicate.
--
-- Schedule: 00:05 UTC every day — five minutes after UTC midnight so the
-- previous day's data is definitely complete before deletion.
-- =============================================================================
DO $cron_setup$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM   pg_extension
        WHERE  extname = 'pg_cron'
    ) THEN
        -- Delete rate-limit rows from before today at 00:05 UTC daily.
        PERFORM cron.schedule(
            'taxbot-ip-rl-daily-cleanup',
            '5 0 * * *',
            $$SELECT private.ip_rl_cleanup(1)$$
        );

        -- Prune pg_cron's own history to the last 7 days to prevent unbounded growth.
        PERFORM cron.schedule(
            'taxbot-cron-history-cleanup',
            '10 0 * * *',
            $$DELETE FROM cron.job_run_details WHERE end_time < now() - INTERVAL '7 days'$$
        );

        RAISE NOTICE 'pg_cron jobs scheduled: taxbot-ip-rl-daily-cleanup, taxbot-cron-history-cleanup';
    ELSE
        RAISE NOTICE 'pg_cron not available — skipping cron job registration. Enable the extension in Supabase → Database → Extensions.';
    END IF;
END;
$cron_setup$;
