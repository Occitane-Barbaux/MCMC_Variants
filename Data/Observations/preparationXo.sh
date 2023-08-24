cdo sellonlatbox,-10,30,35,70 CRUTEM.5.0.1.0.anomalies.nc tmp.nc
cdo fldmean tmp.nc tmp1.nc
cdo yearmean tmp1.nc Xo.nc

cdo sellonlatbox,-10,30,35,70 absolute_v5.nc tmp2.nc
cdo fldmean tmp2.nc  tmp3.nc
cdo yearmean tmp3.nc absolute.nc



