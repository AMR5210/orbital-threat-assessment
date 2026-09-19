-- Replaces notebook Cell 5 (cleaning + encoding), with two corrections
-- to defects present in the original:
--
-- 1. `class` is joined against a versioned seed (asteroid_class_map)
--    instead of `pandas.Categorical.cat.codes`. The original computed
--    the integer code from whichever class values happened to be
--    present in the dataframe at that moment — drop or add a class and
--    every code downstream silently shifts. The seed fixes the codes
--    permanently; see legacy/Project_codeFile.ipynb Cell 5 and
--    dbt/seeds/asteroid_class_map.csv for the exact recovered mapping.
--
-- 2. Rows with a null `neo` flag are dropped, not coerced to 0. The
--    original did `(df['neo'] == 'Y').astype(int)` *before* its
--    blanket dropna(), so a missing neo silently became "confirmed not
--    a NEO." All 4 rows this affects are class=HYA (Hyperbolic
--    Asteroid) — NEO status is genuinely undefined for a body on a
--    hyperbolic trajectory, not just missing. One of the 4 is
--    'Oumuamua (moid=0.096, H=22.08 — a near-miss on both PHA
--    conditions), which is exactly the kind of edge case a silent
--    default should not paper over.
--
-- spkid is retained here (dropped in the original at this same step)
-- so individual predictions stay traceable back to a specific asteroid
-- through the rest of the pipeline. It is dropped at the ML feature
-- boundary instead — see src/otda/data.py.

with source as (
    select * from {{ source('raw', 'asteroids') }}
),

cleaned as (
    select
        spkid,
        (pha = 'Y')                      as is_pha,
        (neo = 'Y')                      as is_neo,
        class,
        H,
        e,
        a,
        q,
        i,
        om,
        w,
        ma,
        ad,
        n,
        per,
        per_y,
        moid,
        rms
    from source
    where
        -- dropna(subset=['pha']) — target must be known
        pha is not null and pha != ''
        -- fixes defect 2 above: neo must be known, not defaulted
        and neo is not null and neo != ''
),

joined as (
    select
        cleaned.*,
        class_map.class_code
    from cleaned
    left join {{ ref('asteroid_class_map') }} as class_map
        on cleaned.class = class_map.class_abbr
)

-- replicates the blanket dropna() cell 5 ends with, across every
-- column that survives into modeling. class_code null (a class value
-- absent from the seed) is caught explicitly by the relationships
-- test in schema.yml rather than silently dropped here.
select *
from joined
where
    H is not null
    and e is not null
    and a is not null
    and q is not null
    and i is not null
    and om is not null
    and w is not null
    and ma is not null
    and ad is not null
    and n is not null
    and per is not null
    and per_y is not null
    and moid is not null
    and rms is not null
    and class_code is not null
