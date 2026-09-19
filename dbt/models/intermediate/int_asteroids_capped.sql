-- Replaces notebook Cell 9 (outlier capping). Caps a, q, moid, i at the
-- 99th percentile computed in int_feature_caps — one definition, shared
-- by both the sample and full-population marts (the original notebook
-- duplicated this clipping logic separately in Cells 9 and 33).

select
    s.spkid,
    s.is_pha,
    s.is_neo,
    s.class,
    s.class_code,
    s.H,
    s.e,
    least(s.a, c.a_p99)       as a,
    least(s.q, c.q_p99)       as q,
    least(s.i, c.i_p99)       as i,
    s.om,
    s.w,
    s.ma,
    s.ad,
    s.n,
    s.per,
    s.per_y,
    least(s.moid, c.moid_p99) as moid,
    s.rms
from {{ ref('stg_asteroids') }} as s
cross join {{ ref('int_feature_caps') }} as c
