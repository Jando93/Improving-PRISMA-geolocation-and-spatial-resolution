This repository contains the two main code blocks constituting the processing workflow for the improvement of geolocation accuracy and spatial resolution of PRISMA hyperspectral imagery (https://www.asi.it/scienze-della-terra/prisma/). It is part of the results of research activity carried out in framework of PRISCAV project by the Institute of BioEconomy (IBE) - National Reasearch Council of Italy (CNR) and represents official public data linked to the scientific paper titled

"_De Luca, G., Carotenuto, F., Genesio, L., Pepe, M., Toscano, P., Boschetti, M., Miglietta, F., Gioli, B. Improving PRISMA hyperspectral spatial resolution and geolocation by using Sentinel2: development and test of an operational procedure in urban and rural areas. ISPRS Journal of Photogrammetry and Remote Sensing. Under Review_". 

In particular, the study aimed at the enhancement of PRISMA spatial resolution by employing the spatial and spectral information of the Sentinel-2 MS data. As first step, the proposed workflow involved the correct sub-pixel alignment of the two datasets. For this purpose, the displacements between the two datasets were detected adopting the AROSICS (Automated and Robust Open-Source Image Co-Registration Software) free-available library (https://github.com/GFZ/arosics), built on a phase correlation criterion. Subsequently, the HySure (HS Super Resolution) fusion method (https://github.com/alfaiate/HySure), in which the fusion process between PRISMA and Sentinel-2 bands is formulated as the resolution of convex quadratic optimization exploiting subspace-based regularization, was employed to obtain a high spectral and spatial resolution HS dataset.

The first code block has been developed via Python v.3.6. 
