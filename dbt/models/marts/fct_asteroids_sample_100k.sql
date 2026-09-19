-- Legacy-parity path: reproduces notebook Cell 6's 100K stratified
-- sample (train_test_split(..., stratify=y, random_state=42)) for
-- comparing against the originally published README numbers.
--
-- NOT a bit-for-bit reproduction — sklearn's Mersenne Twister and
-- DuckDB's hash() are unrelated PRNGs, so this cannot select the exact
-- same 100,000 rows. What IS reproduced is the method: same size
-- (100,000), same stratification by is_pha, same resulting ratio
-- (2,066 / 932,335 = 0.2216% population ratio -> 222 positives at
-- 100K, matching Cell 6's printed "PHA=1 in sample: 222 (0.222%)").
--
-- hash(spkid || salt) gives a stable, deterministic ordering per
-- stratum without relying on DuckDB's session-scoped setseed() pragma
-- (which would need a pre-hook and isn't guaranteed to share a
-- connection with this query under threaded runs). This is DuckDB-
-- specific SQL (hash()); revisit if this model is ever pointed at the
-- Snowflake target.

with capped as (
    select * from {{ ref('int_asteroids_capped') }}
),

ranked as (
    select
        capped.*,
        row_number() over (
            partition by is_pha
            order by hash(spkid || '_otda_sample_seed_42')
        ) as rn
    from capped
),

target_counts as (
    select
        cast(round(100000.0 * sum(case when is_pha then 1 else 0 end) / count(*)) as bigint) as n_pha
    from capped
)

select
    ranked.spkid,
    ranked.is_pha,
    ranked.is_neo,
    ranked.class,
    ranked.class_code,
    ranked.H,
    ranked.e,
    ranked.a,
    ranked.q,
    ranked.i,
    ranked.per,
    ranked.moid
from ranked
cross join target_counts
where
    (ranked.is_pha and ranked.rn <= target_counts.n_pha)
    or (not ranked.is_pha and ranked.rn <= (100000 - target_counts.n_pha))
