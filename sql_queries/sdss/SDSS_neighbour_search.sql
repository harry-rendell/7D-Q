-- This query takes the DR14Q coords and finds the nearest associated objects within 3"
-- The output is NOT used to find secondary observations, it is just used as a check to see the kinds of objects are nearby.

-- qsos
select

q.uid as uid, p.objID, 
p.ra as ra, p.dec as dec,
q.ra as ra_ref, q.dec as dec_ref,
s.class as class,
nb.distance as get_nearby_distance

into mydb.sdss_neighbours_qsos
from mydb.qsos_subsample_coords
cross apply dbo.fGetNearbyObjEq(q.ra, q.dec, 0.05) as nb

join PhotoPrimary p on p.objid=nb.objid
join specobj as s on s.bestobjid=p.objid

-- calibStars
select

q.uid_s as uid_s, p.objID, 
p.ra as ra, p.dec as dec,
q.ra as ra_ref, q.dec as dec_ref,
s.class as class,
nb.distance as get_nearby_distance

into mydb.sdss_neighbours_calibStars
from mydb.calibStars_subsample_coords
cross apply dbo.fGetNearbyObjEq(q.ra, q.dec, 0.05) as nb

join PhotoPrimary p on p.objid=nb.objid
join specobj as s on s.bestobjid=p.objid
