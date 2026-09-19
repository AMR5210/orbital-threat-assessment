-- Same invariant, checked on the 100K legacy-parity sample. The
-- sampling logic in fct_asteroids_sample_100k targets this ratio by
-- construction (target_counts CTE) — this test guards against that
-- construction breaking silently in a future edit.

with ratio as (
    select
        sum(case when is_pha then 1 else 0 end)::double / count(*) as pha_ratio
    from {{ ref('fct_asteroids_sample_100k') }}
)

select *
from ratio
where pha_ratio < 0.0020 or pha_ratio > 0.0025
