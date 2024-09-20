-- This query takes objID_ps from PS_neighbour_search and finds all observations associated with those IDs.
-- Unlike SDSS, there is enough allocated space to run this in one go
-- result should be saved as data/surveys/sdss/{OBJ}/sdss_secondary.csv

-- qsos
select
    q.uid, d.objID, d.detectID, filter=f.filterType, d.obsTime, d.ra, d.dec,
    d.psfFlux, d.psfFluxErr
into mydb.ps_secondary
from mydb.ps_neighbours_qsos q
join Detection d on d.objid = q.objid_ps
join Filter f on f.filterID = d.filterID
order by uid ASC, obsTime ASC

-- calibStars
select
    q.uid_s, d.objID, d.detectID, filter=f.filterType, d.obsTime, d.ra, d.dec,
    d.psfFlux, d.psfFluxErr
into mydb.ps_secondary
from mydb.ps_neighbours_calibStars q
join Detection d on d.objid = q.objid_ps
join Filter f on f.filterID = d.filterID
order by uid_s ASC, obsTime ASC
