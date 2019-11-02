#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib as mt
#from matplotlib.lines import Line2D
import numpy.polynomial.polynomial as poly
import glob
import gc
from scipy import stats
from lmfit import Model
from lmfit.models import LinearModel, LorentzianModel,GaussianModel,SkewedGaussianModel,ExponentialGaussianModel,ExponentialModel,PolynomialModel





#File_Directory = '/Volumes/My Book/scripts/Analysis/Python_Analysis/Data_Files/Cherenkov_Calibration/H3/kin16'
File_Directory = '/Users/michaelnycz/Documents/Pass2_Analysis/Cherenkov_Calibration/H3/kin16'



H3_List = []
H3_List.append(glob.glob(File_Directory + '/Run_*.csv'))



H3_List.sort()


# In[6]:


len(H3_List[0])


# In[7]:


H3_List[0]


# In[8]:


H3=[]
for i in range(0,len(H3_List[0])):
    H3.append(pd.read_csv(H3_List[0][i]))


# In[9]:


H3_Total=[]
#for i in range(0,len(H3)):
    #H3_Total.append(H3[i])
H3_Total=pd.concat([H3[0],H3[1],H3[2],H3[3],H3[4],H3[5],H3[6],H3[7],H3[8],H3[9],H3[10],H3[11],H3[12],H3[13]],axis=0,ignore_index=True)    


# In[38]:


H3_Total.tail()


# In[ ]:





# In[12]:


plt.hist(H3_Total['Cher_Un_9'],bins=100,histtype='step',range=[100,1000])
plt.hist(H3_Total['Cher_Cor_9'],bins=100,histtype='step',range=[100,1000])
plt.axvline(300,0,1)
plt.yscale('log')



# In[13]:


Cher = H3_Total.columns.get_values()


# In[14]:



y_data=[[],]*10
x_data=[[],]*10
x_boarder = [[],]*10
for i in range(0,10):
    y_data[i],x_boarder[i],_ = plt.hist(H3_Total[Cher[i]],bins=70,range=[300,600],histtype='step',linewidth=3)
    x_data[i] = x_boarder[i][:-1] + np.diff(x_boarder[i]) / 2
plt.rcParams["figure.figsize"] =[40,30]
plt.rc('xtick',labelsize=40)
plt.rc('ytick',labelsize=40)   


# In[ ]:





# In[18]:


Gaussian=GaussianModel(prefix='gaus_')
#pars_gaus=Gaussian.guess(y_data,x=x_data)



pars_gaus=[[]]*10
for i in range(0,len(x_data)):
    pars_gaus[i]=Gaussian.guess(y_data[i],x=x_data[i])



Gaussian_output=[[]]*10
for i in range(0,len(x_data)):
    Gaussian_output[i] = Gaussian.fit(y_data[i],pars_gaus[i],x=x_data[i])



Gauss_Mean = [[]]*10
Gauss_Sigma = [[]]*10
Gauss_Amp = [[]]*10

for i in range(0,len(x_data)):
    Gauss_Mean[i] = Gaussian_output[i].params['gaus_center'].value
    Gauss_Sigma[i] = Gaussian_output[i].params['gaus_sigma'].value
    Gauss_Amp[i] = Gaussian_output[i].params['gaus_amplitude'].value


# In[22]:


Gauss_Mean[1],Gauss_Sigma[0]


# In[23]:


for i in range(0,2):
    print(Gaussian_output[i].params['gaus_center'].value)


# In[24]:


print(Gaussian_output[1].params['gaus_center'].value),print(Gaussian_output[1].params['gaus_sigma'].value)


# In[43]:


fig0=plt
for i in range(0,10):
    plt.subplot(5,2,i+1)
    plt.hist(H3_Total[Cher[i]],bins=70,histtype='step',range=[100,600],linewidth=6,label='Cher %d Uncalibrated'%i)
    plt.hist(H3_Total[Cher[i+10]],bins=70,histtype='step',range=[100,600],linewidth=6,label='Cher %d Old Calibration'%i)
    plt.axvline(300,0,1,color='black',linestyle='--',linewidth=5,label='Proposed mean of SPE Peak')
    plt.rcParams["figure.figsize"] =[70,60] 
    plt.rc('xtick',labelsize=40)
    plt.rc('ytick',labelsize=40)   
    plt.legend(loc=2,fontsize=40)
#fig0.savefig('/Users/michaelnycz/Documents/Pass2_Analysis/Images/Final_Ratio_Images/Cherenkov_Calib/Comparison_plot.png',fmt='png')


# In[42]:


fig1=plt
fig1.suptitle('Fit with Gaussian',fontsize=80)
for i in range(0,8):
    #plt.subplot(5,2,i+1)
    plt.subplot(4,2,i+1)
    plt.hist(H3_Total[Cher[i]],bins=70,histtype='step',range=[300,600],linewidth=6,label='Cher %d'%i)
    plt.plot(x_data[i],Gaussian_output[i].best_fit,'g-',linewidth=5,label='Gaussian fit')
    #plt.axvline(300,0,1,color='black',linestyle='--',linewidth=5)
    plt.rcParams["figure.figsize"] =[50,40]
    plt.rc('xtick',labelsize=40)
    plt.rc('ytick',labelsize=40)   
    plt.legend(loc=1,fontsize=40)

#fig1.savefig('/Users/michaelnycz/Documents/Pass2_Analysis/Images/Final_Ratio_Images/Cherenkov_Calib/Gaus_fit.png',fmt='png')


# In[26]:


Gain = []
def Gain_Factor(x):
    return(300/x)

Gain = np.vectorize(Gain_Factor)(Gauss_Mean)


# In[41]:


Gain


# In[44]:


H3_Total


# In[29]:


H3_Corrected = [[]]*10
#def Apply_Gain(x,y):
    #return (x*y)

for i in range(0,10):
    def Apply_Gain(x,y):
        return (x*y)
    H3_Corrected[i] = np.vectorize(Apply_Gain)(Gain[i],H3_Total[Cher[i]])
    H3_Corrected[i] =pd.DataFrame(H3_Corrected[i],columns=[Cher[i+10]])

H3_Total_Corrected = pd.concat(H3_Corrected,axis=1)




H3_Total_test = H3_Total.copy()


# In[31]:


H3_Total.columns


# In[ ]:





# In[32]:


H3_Total_test.query('Track==1',inplace=True)
H3_Total_test.reset_index(drop=True,inplace=True)


# In[29]:


plt.hist(H3_Total_test['Cher_Sum'],bins=100,range=[0,5000],histtype='step',linewidth=5)
plt.hist(H3_Total['Cher_Sum'],bins=100,range=[0,5000],histtype='step',linewidth=5)
plt.yscale('log')
plt.axvline(300,0,1)


# In[50]:


len(H3_Total),len(H3_Total_Corrected)


# In[37]:


fig3=plt
fig3.suptitle('After Applying New Gain Matched Values',fontsize=80)
for i in range(0,8):
    #plt.subplot(5,2,i+1)
    plt.subplot(4,2,i+1)
    plt.hist(H3_Total_Corrected[Cher[i+10]],bins=70,histtype='step',range=[100,600],linewidth=6,label='Cher %d New Calibration'%i)
    plt.axvline(300,0,1,color='black',linestyle='--',linewidth=6)
    plt.rcParams["figure.figsize"] =[60,50] 
    plt.rc('xtick',labelsize=40)
    plt.rc('ytick',labelsize=40)   
    plt.legend(loc=1,fontsize=40)   
#fig3.savefig('/Users/michaelnycz/Documents/Pass2_Analysis/Images/Final_Ratio_Images/Cherenkov_Calib/New_Gain_Matched.png',fmt='png')



H3_Total_Corrected['Cher_Sum']=H3_Total_Corrected.sum(axis=1)




fig4=plt
plt.hist(H3_Total_Corrected['Cher_Sum'],bins=200,histtype='step',range=[0,3000],color='red',linewidth=8,label='New Calibration')
plt.hist(H3_Total['Cher_Sum'],bins=200,histtype='step',range=[0,3000],linewidth=8,color='blue',label='Old Calibration')
#plt.axvline(2000,0,1,color='black',linestyle='--',linewidth=5,label='Current Cherenkov Sum Cut')
plt.axvline(300,0,1,color='black',linestyle='--',linewidth=8,label='SPE aligned to channel 300')
#plt.axvline(500,0,1,color='blue',linestyle='--',linewidth=5,label='Current Cherenkov Sum Cut')
plt.rcParams["figure.figsize"] =[50,30]
plt.yscale('log')
#plt.title('Cherenkov Sum Old vs Cherenkov Sum New',fontsize=100)
#plt.rcParams["figure.figsize"] =[60,50] 
plt.xlabel('ADC Channels',fontsize=100)
plt.tick_params(which='both',top=True,right=True)
#plt.rc('xtick',labelsize=100)
#plt.rc('ytick',labelsize=100)   
plt.rcParams["axes.linewidth"]  = 6
handle1 = mt.lines.Line2D([], [], c='r',label='New Calibration',linewidth=8)
handle2 = mt.lines.Line2D([], [], c='b',label='Old Calibration',linewidth=8)
legend=plt.legend(handles=[handle1, handle2],fontsize=100,frameon=True)
legend.get_frame().set_edgecolor('black')
plt.minorticks_on()
plt.rcParams['xtick.major.size'] = 40
plt.rcParams['xtick.major.width'] = 7
plt.rcParams['xtick.minor.size'] = 20
plt.rcParams['xtick.minor.width'] = 7
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 40
plt.rcParams['ytick.major.width'] = 7
plt.rcParams['ytick.minor.size'] = 20
plt.rcParams['ytick.minor.width'] = 7
plt.rcParams['ytick.direction'] = 'in'
#fig4.savefig('/Users/michaelnycz/Documents/Pass2_Analysis/Thesis/ThesisImages/Cher_Sum_Comparison.pdf',fmt='pdf')




# In[94]:


y_data_fit=[[],]*10
x_data_fit=[[],]*10
x_boarder_fit = [[],]*10
for i in range(0,10):
    if (i<8):
        y_data_fit[i],x_boarder_fit[i],_ = plt.hist(H3_Total[Cher[i]],bins=40,range=[Gauss_Mean[i]-(Gauss_Sigma[i]/2),Gauss_Mean[i]+(Gauss_Sigma[i]/2)],histtype='step')
        x_data_fit[i] = x_boarder_fit[i][:-1] + np.diff(x_boarder_fit[i]) / 2
    elif(i>7):
        y_data_fit[i],x_boarder_fit[i],_ = plt.hist(H3_Total[Cher[i]],bins=40,range=[Gauss_Mean[i]-180,Gauss_Mean[i]+200],histtype='step')
        x_data_fit[i] = x_boarder_fit[i][:-1] + np.diff(x_boarder_fit[i]) / 2


# In[40]:


(Gauss_Sigma[i]/2)


# In[99]:


for i in range(1,6):
    plt.hist(H3_Total[Cher[i]],bins=40,histtype='step',range=[Gauss_Mean[i]-(Gauss_Sigma[i]/2),Gauss_Mean[i]+(Gauss_Sigma[i]/2)])
    
    
    


# In[57]:


plt.hist(H3_Total[Cher[9]],bins=100,histtype='step',range=[200,800])
plt.yscale('log')


# In[ ]:





# In[11]:


print(Gaussian_output.fit_report(min_correl=0.1))


# In[67]:


plt.hist(H3_Total['Cher_4'],bins=100,histtype='step',range=[200,800])
plt.axvline(300,0,1)
#plt.yscale('log')
plt.plot(x_data[0],Gaussian_output[4].best_fit,'g-',label='Gaussian fit')


# In[ ]:




