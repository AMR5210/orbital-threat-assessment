-- The entire project's validity rests on the ~449:1 class imbalance
-- surviving cleaning intact. Original notebook printed this ratio at
-- four separate points (Cells 5, 6, 16, 27) and relied on someone
-- eyeballing it; this is that check made automatic and warehouse-side.
--
-- Bounds: expected ratio at this stage is 2066 / 932331 = 0.002217
-- (1:451, after the neo-null/HYA fix — see stg_asteroids.sql). Band is
-- [0.0020, 0.0025], generous enough not to be flaky but tight enough
-- that losing a stratify= somewhere would trip it immediately.
--
-- A singular test FAILS if this query returns any rows.

with ratio as (
    select
        sum(case when is_pha then 1 else 0 end)::double / count(*) as pha_ratio
    from {{ ref('stg_asteroids') }}
)

select *
from ratio
where pha_ratio < 0.0020 or pha_ratio > 0.0025
