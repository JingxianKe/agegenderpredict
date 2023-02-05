#!/bin/bash
# Download aligned datasets
wget --user=adiencedb --password=adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.tar.gz
# jieyawenjian
tar -xvf aligned.tar.gz
# Download labels
mkdir folds
cd folds
wget --user=adiencedb --password=adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_0_data.txt
wget --user=adiencedb --password=adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_1_data.txt
wget --user=adiencedb --password=adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_2_data.txt
wget --user=adiencedb --password=adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_3_data.txt
wget --user=adiencedb --password=adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_4_data.txt
