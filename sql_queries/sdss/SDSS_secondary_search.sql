-- This query takes coordinates and finds all secondary observations within 1"
-- Note, there is often not enough allocated space on CasJobs to run this in one go. Therefore we do it in two chunks.
-- The chunks should be concatenated (in bash) and result should be saved as data/surveys/sdss/{OBJ}/sdss_secondary.csv

-- qsos
select

q.uid as uid, p.objID,
q.ra, q.dec,
nb.distance as get_nearby_distance,

p.psfMag_u as mag_u, p.psfMagErr_u as magerr_u,
p.psfMag_g as mag_g, p.psfMagErr_g as magerr_g,
p.psfMag_r as mag_r, p.psfMagErr_r as magerr_r,
p.psfMag_i as mag_i, p.psfMagErr_i as magerr_i,
p.psfMag_z as mag_z, p.psfMagErr_z as magerr_z,
f.mjd_r as mjd

into mydb.sdss_secondary_qsos_0
from mydb.qsos_subsample_coords_0 q
cross apply dbo.fGetNearbyObjAllEq(q.ra, q.dec, 1.0/60.0) as nb

join photoobj p on p.objid=nb.objid
join field f on f.fieldid=p.fieldid
  
ORDER BY uid ASC, mjd ASC

-- calibStars
select

q.uid_s as uid_s, p.objID,
q.ra, q.dec,
nb.distance as get_nearby_distance,

p.psfMag_u as mag_u, p.psfMagErr_u as magerr_u,
p.psfMag_g as mag_g, p.psfMagErr_g as magerr_g,
p.psfMag_r as mag_r, p.psfMagErr_r as magerr_r,
p.psfMag_i as mag_i, p.psfMagErr_i as magerr_i,
p.psfMag_z as mag_z, p.psfMagErr_z as magerr_z,
f.mjd_r as mjd

into mydb.sdss_secondary_calibStars_0
from mydb.calibStars_subsample_coords_0 q
cross apply dbo.fGetNearbyObjAllEq(q.ra, q.dec, 1.0/60.0) as nb

join photoobj p on p.objid=nb.objid
join field f on f.fieldid=p.fieldid
  
ORDER BY uid_s ASC, mjd ASC
