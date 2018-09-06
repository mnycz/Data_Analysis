#! /Users/michaelnycz/anaconda3/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import os
import math
File = '/w/halla-scifs17exp/triton/mnycz/Marathon/Analysis/Xsec/Beta_Decay_Correction'
#File = '/Volumes/My Book/scripts/Analysis/Xsec/Beta_Decay_Correction'
#Rad_Cor = '/Volumes/My Book/scripts/Analysis/Xsec/Rad_Cor'
Rad_Cor  = '/w/halla-scifs17exp/triton/mnycz/Marathon/Analysis/Xsec/Rad_Cor'
Error = '/w/halla-scifs17exp/triton/mnycz/Marathon/Analysis/python'
#H3 = pd.DataFrame()
#nbins = 50

# read in Yields
H3_1=pd.read_table('%s/H3_kin1_Yield.txt' %File ,delim_whitespace=True,names=('Yield_1','x_bj_1'))
H3_2=pd.read_table('%s/H3_kin2_Yield.txt' %File ,delim_whitespace=True,names=('Yield_2','x_bj_2'))
H3_3=pd.read_table('%s/H3_kin3_Yield.txt' %File ,delim_whitespace=True,names=('Yield_3','x_bj_3'))
H3_5=pd.read_table('%s/H3_kin5_Yield.txt' %File ,delim_whitespace=True,names=('Yield_5','x_bj_5'))
H3_7=pd.read_table('%s/H3_kin7_Yield.txt' %File ,delim_whitespace=True,names=('Yield_7','x_bj_7'))
H3_9=pd.read_table('%s/H3_kin9_Yield.txt' %File ,delim_whitespace=True,names=('Yield_9','x_bj_9'))
H3_11=pd.read_table('%s/H3_kin11_Yield.txt' %File ,delim_whitespace=True,names=('Yield_11','x_bj_11'))
H3_13=pd.read_table('%s/H3_kin13_Yield.txt' %File ,delim_whitespace=True,names=('Yield_13','x_bj_13'))
H3_15=pd.read_table('%s/H3_kin15_Yield.txt' %File ,delim_whitespace=True,names=('Yield_15','x_bj_15'))

D2_1=pd.read_table('%s/D2_kin1_Yield.txt' %File ,delim_whitespace=True,names=('Yield_1','x_bj_1'))
D2_2=pd.read_table('%s/D2_kin2_Yield.txt' %File ,delim_whitespace=True,names=('Yield_2','x_bj_2'))
D2_3=pd.read_table('%s/D2_kin3_Yield.txt' %File ,delim_whitespace=True,names=('Yield_3','x_bj_3'))
D2_5=pd.read_table('%s/D2_kin5_Yield.txt' %File ,delim_whitespace=True,names=('Yield_5','x_bj_5'))
D2_7=pd.read_table('%s/D2_kin7_Yield.txt' %File ,delim_whitespace=True,names=('Yield_7','x_bj_7'))
D2_9=pd.read_table('%s/D2_kin9_Yield.txt' %File ,delim_whitespace=True,names=('Yield_9','x_bj_9'))
D2_11=pd.read_table('%s/D2_kin11_Yield.txt' %File ,delim_whitespace=True,names=('Yield_11','x_bj_11'))
D2_13=pd.read_table('%s/D2_kin13_Yield.txt' %File ,delim_whitespace=True,names=('Yield_13','x_bj_13'))
D2_15=pd.read_table('%s/D2_kin15_Yield.txt' %File ,delim_whitespace=True,names=('Yield_15','x_bj_15'))

# read in Rad Cor 
H3_R_1=pd.read_table('%s/H3_bin_xs_1.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_1','Born_1','Rad_1'))
H3_R_2=pd.read_table('%s/H3_bin_xs_2.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_2','Born_2','Rad_2'))
H3_R_3=pd.read_table('%s/H3_bin_xs_3.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_3','Born_3','Rad_3'))
H3_R_5=pd.read_table('%s/H3_bin_xs_5.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_5','Born_5','Rad_5'))
H3_R_7=pd.read_table('%s/H3_bin_xs_7.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_7','Born_7','Rad_7'))
H3_R_9=pd.read_table('%s/H3_bin_xs_9.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_9','Born_9','Rad_9'))
H3_R_11=pd.read_table('%s/H3_bin_xs_11.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_11','Born_11','Rad_11'))
H3_R_13=pd.read_table('%s/H3_bin_xs_13.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_13','Born_13','Rad_13'))
H3_R_15=pd.read_table('%s/H3_bin_xs_15.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_15','Born_15','Rad_15'))

D2_R_1=pd.read_table('%s/D2_bin_xs_1.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_1','Born_1','Rad_1'))
D2_R_2=pd.read_table('%s/D2_bin_xs_2.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_2','Born_2','Rad_2'))
D2_R_3=pd.read_table('%s/D2_bin_xs_3.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_3','Born_3','Rad_3'))
D2_R_5=pd.read_table('%s/D2_bin_xs_5.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_5','Born_5','Rad_5'))
D2_R_7=pd.read_table('%s/D2_bin_xs_7.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_7','Born_7','Rad_7'))
D2_R_9=pd.read_table('%s/D2_bin_xs_9.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_9','Born_9','Rad_9'))
D2_R_11=pd.read_table('%s/D2_bin_xs_11.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_11','Born_11','Rad_11'))
D2_R_13=pd.read_table('%s/D2_bin_xs_13.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_13','Born_13','Rad_13'))
D2_R_15=pd.read_table('%s/D2_bin_xs_15.out' %Rad_Cor ,delim_whitespace=True,names=('x_bj_15','Born_15','Rad_15'))

#ERROR
H3_ER_1=pd.read_table('%s/H3_kin1_Stat.txt' %Error ,delim_whitespace=True,names=('Error_1','x_bj_1'))
H3_ER_2=pd.read_table('%s/H3_kin2_Stat.txt' %Error ,delim_whitespace=True,names=('Error_2','x_bj_2'))
H3_ER_3=pd.read_table('%s/H3_kin3_Stat.txt' %Error ,delim_whitespace=True,names=('Error_3','x_bj_3'))
H3_ER_5=pd.read_table('%s/H3_kin5_Stat.txt' %Error ,delim_whitespace=True,names=('Error_5','x_bj_5'))
H3_ER_7=pd.read_table('%s/H3_kin7_Stat.txt' %Error ,delim_whitespace=True,names=('Error_7','x_bj_7'))
H3_ER_9=pd.read_table('%s/H3_kin9_Stat.txt' %Error ,delim_whitespace=True,names=('Error_9','x_bj_9'))
H3_ER_11=pd.read_table('%s/H3_kin11_Stat.txt' %Error ,delim_whitespace=True,names=('Error_11','x_bj_11'))
H3_ER_13=pd.read_table('%s/H3_kin13_Stat.txt' %Error ,delim_whitespace=True,names=('Error_13','x_bj_13'))
H3_ER_15=pd.read_table('%s/H3_kin15_Stat.txt' %Error ,delim_whitespace=True,names=('Error_15','x_bj_15'))

D2_ER_1=pd.read_table('%s/D2_kin1_Stat.txt' %Error ,delim_whitespace=True,names=('Error_1','x_bj_1'))
D2_ER_2=pd.read_table('%s/D2_kin2_Stat.txt' %Error ,delim_whitespace=True,names=('Error_2','x_bj_2'))
D2_ER_3=pd.read_table('%s/D2_kin3_Stat.txt' %Error ,delim_whitespace=True,names=('Error_3','x_bj_3'))
D2_ER_5=pd.read_table('%s/D2_kin5_Stat.txt' %Error ,delim_whitespace=True,names=('Error_5','x_bj_5'))
D2_ER_7=pd.read_table('%s/D2_kin7_Stat.txt' %Error ,delim_whitespace=True,names=('Error_7','x_bj_7'))
D2_ER_9=pd.read_table('%s/D2_kin9_Stat.txt' %Error ,delim_whitespace=True,names=('Error_9','x_bj_9'))
D2_ER_11=pd.read_table('%s/D2_kin11_Stat.txt' %Error ,delim_whitespace=True,names=('Error_11','x_bj_11'))
D2_ER_13=pd.read_table('%s/D2_kin13_Stat.txt' %Error ,delim_whitespace=True,names=('Error_13','x_bj_13'))
D2_ER_15=pd.read_table('%s/D2_kin15_Stat.txt' %Error ,delim_whitespace=True,names=('Error_15','x_bj_15'))



#H3 = pd.concat([H3_1,H3_2],axis=1)
H3_1_h = plt.hist(H3_1.x_bj_1,50,range=[0,1],weights = H3_1.Yield_1)
D2_1_h = plt.hist(D2_1.x_bj_1,50,range=[0,1],weights = D2_1.Yield_1)
H3_2_h = plt.hist(H3_2.x_bj_2,50,range=[0,1],weights = H3_2.Yield_2)
D2_2_h = plt.hist(D2_2.x_bj_2,50,range=[0,1],weights = D2_2.Yield_2)
H3_3_h = plt.hist(H3_3.x_bj_3,50,range=[0,1],weights = H3_3.Yield_3)
D2_3_h = plt.hist(D2_3.x_bj_3,50,range=[0,1],weights = D2_3.Yield_3)
H3_5_h = plt.hist(H3_5.x_bj_5,50,range=[0,1],weights = H3_5.Yield_5)
D2_5_h = plt.hist(D2_5.x_bj_5,50,range=[0,1],weights = D2_5.Yield_5)
H3_7_h = plt.hist(H3_7.x_bj_7,50,range=[0,1],weights = H3_7.Yield_7)
D2_7_h = plt.hist(D2_7.x_bj_7,50,range=[0,1],weights = D2_7.Yield_7)
H3_9_h = plt.hist(H3_9.x_bj_9,50,range=[0,1],weights = H3_9.Yield_9)
D2_9_h = plt.hist(D2_9.x_bj_9,50,range=[0,1],weights = D2_9.Yield_9)
H3_11_h = plt.hist(H3_11.x_bj_11,50,range=[0,1],weights = H3_11.Yield_11)
D2_11_h = plt.hist(D2_11.x_bj_11,50,range=[0,1],weights = D2_11.Yield_11)
H3_13_h = plt.hist(H3_13.x_bj_13,50,range=[0,1],weights = H3_13.Yield_13)
D2_13_h = plt.hist(D2_13.x_bj_13,50,range=[0,1],weights = D2_13.Yield_13)
H3_15_h = plt.hist(H3_15.x_bj_15,50,range=[0,1],weights = H3_15.Yield_15)
D2_15_h = plt.hist(D2_15.x_bj_15,50,range=[0,1],weights = D2_15.Yield_15)


H3_1_Y = pd.DataFrame({"Y_1":H3_1_h[0]})
D2_1_Y = pd.DataFrame({"Y_1":D2_1_h[0]})
H3_2_Y = pd.DataFrame({"Y_2":H3_2_h[0]})
D2_2_Y = pd.DataFrame({"Y_2":D2_2_h[0]})
H3_3_Y = pd.DataFrame({"Y_3":H3_3_h[0]})
D2_3_Y = pd.DataFrame({"Y_3":D2_3_h[0]})
H3_5_Y = pd.DataFrame({"Y_5":H3_5_h[0]})
D2_5_Y = pd.DataFrame({"Y_5":D2_5_h[0]})
H3_7_Y = pd.DataFrame({"Y_7":H3_7_h[0]})
D2_7_Y = pd.DataFrame({"Y_7":D2_7_h[0]})
H3_9_Y = pd.DataFrame({"Y_9":H3_9_h[0]})
D2_9_Y = pd.DataFrame({"Y_9":D2_9_h[0]})
H3_11_Y = pd.DataFrame({"Y_11":H3_11_h[0]})
D2_11_Y = pd.DataFrame({"Y_11":D2_11_h[0]})
H3_13_Y = pd.DataFrame({"Y_13":H3_13_h[0]})
D2_13_Y = pd.DataFrame({"Y_13":D2_13_h[0]})
H3_15_Y = pd.DataFrame({"Y_15":H3_13_h[0]})
D2_15_Y = pd.DataFrame({"Y_15":D2_13_h[0]})


#Rad_Cor plots
H3_1_r = plt.hist(H3_R_1.x_bj_1,50,range=[0,1],weights = (H3_R_1.Born_1/H3_R_1.Rad_1))
D2_1_r = plt.hist(D2_R_1.x_bj_1,50,range=[0,1],weights = (D2_R_1.Born_1/D2_R_1.Rad_1))
H3_2_r = plt.hist(H3_R_2.x_bj_2,50,range=[0,1],weights = (H3_R_2.Born_2/H3_R_2.Rad_2))
D2_2_r = plt.hist(D2_R_2.x_bj_2,50,range=[0,1],weights = (D2_R_2.Born_2/D2_R_2.Rad_2))
H3_3_r = plt.hist(H3_R_3.x_bj_3,50,range=[0,1],weights = (H3_R_3.Born_3/H3_R_3.Rad_3))
D2_3_r = plt.hist(D2_R_3.x_bj_3,50,range=[0,1],weights = (D2_R_3.Born_3/D2_R_3.Rad_3))
H3_5_r = plt.hist(H3_R_5.x_bj_5,50,range=[0,1],weights = (H3_R_5.Born_5/H3_R_5.Rad_5))
D2_5_r = plt.hist(D2_R_5.x_bj_5,50,range=[0,1],weights = (D2_R_5.Born_5/D2_R_5.Rad_5))
H3_7_r = plt.hist(H3_R_7.x_bj_7,50,range=[0,1],weights = (H3_R_7.Born_7/H3_R_7.Rad_7))
D2_7_r = plt.hist(D2_R_7.x_bj_7,50,range=[0,1],weights = (D2_R_7.Born_7/D2_R_7.Rad_7))
H3_9_r = plt.hist(H3_R_9.x_bj_9,50,range=[0,1],weights = (H3_R_9.Born_9/H3_R_9.Rad_9))
D2_9_r = plt.hist(D2_R_9.x_bj_9,50,range=[0,1],weights = (D2_R_9.Born_9/D2_R_9.Rad_9))
H3_11_r = plt.hist(H3_R_11.x_bj_11,50,range=[0,1],weights = (H3_R_11.Born_11/H3_R_11.Rad_11))
D2_11_r = plt.hist(D2_R_11.x_bj_11,50,range=[0,1],weights = (D2_R_11.Born_11/D2_R_11.Rad_11))
H3_13_r = plt.hist(H3_R_13.x_bj_13,50,range=[0,1],weights = (H3_R_13.Born_13/H3_R_13.Rad_13))
D2_13_r = plt.hist(D2_R_13.x_bj_13,50,range=[0,1],weights = (D2_R_13.Born_13/D2_R_13.Rad_13))
H3_15_r = plt.hist(H3_R_15.x_bj_15,50,range=[0,1],weights = (H3_R_15.Born_15/H3_R_15.Rad_15))
D2_15_r = plt.hist(D2_R_15.x_bj_15,50,range=[0,1],weights = (D2_R_15.Born_15/D2_R_15.Rad_15))


H3_1_rad = pd.DataFrame({"Y_1":H3_1_r[0]})
D2_1_rad = pd.DataFrame({"Y_1":D2_1_r[0]})
H3_2_rad = pd.DataFrame({"Y_2":H3_2_r[0]})
D2_2_rad = pd.DataFrame({"Y_2":D2_2_r[0]})
H3_3_rad = pd.DataFrame({"Y_3":H3_3_r[0]})
D2_3_rad = pd.DataFrame({"Y_3":D2_3_r[0]})
H3_5_rad = pd.DataFrame({"Y_5":H3_5_r[0]})
D2_5_rad = pd.DataFrame({"Y_5":D2_5_r[0]})
H3_7_rad = pd.DataFrame({"Y_7":H3_7_r[0]})
D2_7_rad = pd.DataFrame({"Y_7":D2_7_r[0]})
H3_9_rad = pd.DataFrame({"Y_9":H3_9_r[0]})
D2_9_rad = pd.DataFrame({"Y_9":D2_9_r[0]})
H3_11_rad = pd.DataFrame({"Y_11":H3_11_r[0]})
D2_11_rad = pd.DataFrame({"Y_11":D2_11_r[0]})
H3_13_rad = pd.DataFrame({"Y_13":H3_13_r[0]})
D2_13_rad = pd.DataFrame({"Y_13":D2_13_r[0]})
H3_15_rad = pd.DataFrame({"Y_15":H3_15_r[0]})
D2_15_rad = pd.DataFrame({"Y_15":D2_15_r[0]})

#
bins=np.linspace(0,1,51)
bin_center=[0]*50
for i in range(len(bin_center)):
    bin_center[i]=(bins[i]+bins[i+1])/2
bin_center

ratio1 = (H3_1_Y*H3_1_rad)/(D2_1_Y*D2_1_rad)
ratio2 = (H3_2_Y*H3_2_rad)/(D2_2_Y*D2_2_rad)
ratio3 = (H3_3_Y*H3_3_rad)/(D2_3_Y*D2_3_rad)
ratio5 = (H3_5_Y*H3_5_rad)/(D2_5_Y*D2_5_rad)
ratio7 = (H3_7_Y*H3_7_rad)/(D2_7_Y*D2_7_rad)
ratio9 = (H3_9_Y*H3_9_rad)/(D2_9_Y*D2_9_rad)
ratio11 = (H3_11_Y*H3_11_rad)/(D2_11_Y*D2_11_rad)
ratio13 = (H3_13_Y*H3_13_rad)/(D2_13_Y*D2_13_rad)
ratio15 = (H3_15_Y*H3_15_rad)/(D2_15_Y*D2_15_rad)

ratio1[np.isinf(ratio1)] = 0
ratio1[ ratio1==0 ] = np.nan
ratio2[np.isinf(ratio2)] = 0
ratio2[ ratio2==0 ] = np.nan
ratio3[np.isinf(ratio3)] = 0
ratio3[ ratio3==0 ] = np.nan
ratio5[np.isinf(ratio5)] = 0
ratio5[ ratio5==0 ] = np.nan
ratio7[np.isinf(ratio7)] = 0
ratio7[ ratio7==0 ] = np.nan
ratio9[np.isinf(ratio9)] = 0
ratio9[ ratio9==0 ] = np.nan
ratio11[np.isinf(ratio11)] = 0
ratio11[ ratio11==0 ] = np.nan
ratio13[np.isinf(ratio13)] = 0
ratio13[ ratio13==0 ] = np.nan

plt.ylim(0.6,1.5)
plt.errorbar(bin_center,ratio1,yerr=(H3_ER_1.Error_1/D2_ER_1.Error_1),fmt='o')
plt.errorbar(bin_center,ratio2,fmt='o')
plt.errorbar(bin_center,ratio3,fmt='o')
plt.errorbar(bin_center,ratio5,fmt='o')
plt.errorbar(bin_center,ratio7,fmt='o')
plt.errorbar(bin_center,ratio9,fmt='o')
plt.errorbar(bin_center,ratio11,fmt='o')
plt.errorbar(bin_center,ratio13,fmt='o')

plt.show()
    




