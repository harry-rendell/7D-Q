import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from urllib.error import HTTPError

def format_fname(x,y,z):
	return '{}_band/lc_{}.csv'.format(x, str(int(y)).zfill(len(str(z))))

def request_and_save_photometry(oid_batch, suffix):
	base_url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768"
	url = base_url.format(''.join(['&ID='+str(oid) for oid in oid_batch['oid']])[1:])
	df = pd.read_csv(url, usecols = ['oid','mjd','mag','magerr','filtercode','clrcoeff','limitmag'], dtype=cfg.COLLECTION.ZTF.dtypes)
	df = df.merge(oid_batch, on='oid', how='left', sort=False).sort_values([ID,'mjd'])
	for b in 'gri':
		df[df['filtercode']=='z'+b].to_csv(output_dir + format_fname(b, suffix, n_workers),
										   header=False, 
										   index=False, 
										   mode='a',
										   columns=SAVE_COLS)

def split_save(oid_batch, suffix, n_chunks):
	# split into n chunks and save aindividually
	for oid_batch_halved in np.array_split(oid_batch, n_chunks):
		request_and_save_photometry(oid_batch_halved, suffix)

def obtain_ztf_lightcurves(suffix):
	oids = np.array_split(pd.read_csv(ztf_oids_fname, usecols=[ID,'oid'], dtype=np.uint64), n_workers)[suffix]
	for i, oid_batch in enumerate(np.array_split(oids, n_requests)):
		for n_chunks in [1,2,4,8]:
			try:
				print('batch {}, {}/{} requesting     in {} chunk(s)'.format(suffix+1, i+1, n_requests, n_chunks), flush=True)
				split_save(oid_batch, suffix, n_chunks)
				print('batch {}, {}/{} success saving in {} chunk(s)'.format(suffix+1, i+1, n_requests, n_chunks), flush=True)
				success = True
				break
			except HTTPError as e:
				print('batch {}, {}/{} HTTPError      in {} chunk(s)'.format(suffix+1, i+1, n_requests, n_chunks), flush=True)
				success = False
		if not success:
			print('ERROR: batch {}, {}/{} - Unable to save despite splitting into 8 chunks'.format(suffix+1, i+1, n_requests), flush=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
	parser.add_argument("--n_workers", type=int, required=True, help="Number of workers to use")
	args = parser.parse_args()
	print('args:',args)

	OBJ = args.object
	ID = 'uid' if (OBJ == 'qsos') else 'uid_s'
	SAVE_COLS = [ID,'oid','mjd','mag','magerr','clrcoeff','limitmag']

	total_requests = 6000
	n_workers  = args.n_workers
	n_requests = total_requests // n_workers

	ztf_oids_fname  = os.path.join(cfg.D_DIR, 'surveys/ztf/{}/ztf_oids.csv'.format(OBJ))
	output_dir      = os.path.join(cfg.D_DIR, 'surveys/ztf/{}/dr6'.format(OBJ), '')

	# clear previous saved lightcurves and write header
	for band in 'gri':
		os.makedirs(os.path.join(output_dir, band+'_band'), exist_ok=True)
		for i in range(n_workers):
			with open(os.path.join(output_dir,format_fname(band, i, n_workers)), 'w') as file:
				file.write(','.join(SAVE_COLS) + '\n')

	start = time.time()

	with Pool(n_workers) as pool:
		pool.map(obtain_ztf_lightcurves,range(n_workers))

	print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
