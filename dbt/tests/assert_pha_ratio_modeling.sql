-- Same invariant as assert_pha_ratio_population.sql, checked again at
-- the mart the ML pipeline actually reads from. Capping and column-
-- pruning between stg_asteroids and here should not touch row count
-- or pha values — if this ever disagrees with the staging-level test,
-- something in int_asteroids_capped / int_feature_caps is dropping or
-- duplicating rows.

with ratio as (
    select
        sum(case when is_pha then 1 else 0 end)::double / count(*) as pha_ratio
    from {{ ref('fct_asteroids_modeling') }}
)

select *
from ratio
where pha_ratio < 0.0020 or pha_ratio > 0.0025
