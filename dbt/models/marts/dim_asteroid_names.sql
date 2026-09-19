-- Lookup-only dimension for the Stage 7 agent's fetch_asteroid tool.
--
-- full_name/name/pdes are intentionally absent from fct_asteroids_modeling
-- (dropped at stg_asteroids — never a model feature, per that model's own
-- comments), but an agent answering "is Apophis hazardous?" needs some
-- way to resolve a natural-language reference to a specific spkid. This
-- model exists purely for that lookup; it is never joined into anything
-- ML-facing.
--
-- Inner-joined to fct_asteroids_modeling so every row here is guaranteed
-- to have a corresponding modeling row — no dangling lookups that
-- resolve to a name but then fail prediction.

select
    raw.spkid,
    raw.full_name,
    raw.name,
    raw.pdes
from {{ source('raw', 'asteroids') }} as raw
inner join {{ ref('fct_asteroids_modeling') }} as modeling
    on raw.spkid = modeling.spkid
