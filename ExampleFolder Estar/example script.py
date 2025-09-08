
# make sure to have already installed Estar package, 
# you can use !pip install Estar==1.6.6 to do so

from estar import*
rd.seed(154784.58)


hsfolder ="C:/Users/hoare/OneDrive/Bureau/ExampleFolder Estar/HS"
obsfolder = "C:/Users/hoare/OneDrive/Bureau/ExampleFolder Estar/Obs"
IUCNpath = "C:/Users/hoare/OneDrive/Bureau/ExampleFolder Estar/Reference range"
savefigfolder = "C:/Users/hoare/OneDrive/Bureau/ExampleFolder Estar/Output"

runoverfile(hsfolder,obsfolder,obstaxafile=None,subregionfile=None,savefigfolder=savefigfolder,
            RefRangeDistSmooth=50,WE=0.5,Wdens=0.5,typerange=IUCNpath,
            outputtype="CR + Cut50",plot=False,HStreatment="nanmax",maxpoints=5000,KDE_mode="network KDE",
            listnamesformat=["XxX_count_example.tif","Habitat_Suitability_XxX_example.tif","XxX_IUCN_example.tif"])