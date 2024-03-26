---
title: "Final Project for GPH 2338 Machine Learning in Publich Health"
author: "Olivia Chien and Yucheng Wang"
date: "2024-03-25"
output: 
  html_document:
    keep_md: true
---



## Data download


```r
# download data from ICPSR to local directory
library(icpsrdata)
icpsr_download(file_id = 37516)
```


```r
# import downloaded data
load(file='icpsr_data/ICPSR_37516/DS0001/37516-0001-Data.rda')
raw <- da37516.0001
```

## Data cleaning


```r
# Drop missing data, sample size=4,090
data <- raw[complete.cases(raw[ , c("SMOKER_CATEGORY","ALCOHOL","TRACT_BELOW_POVERTY","TRACT_UNINSURED_OVER_65","TRACT_OWNER_OVER_30","TRACT_UNINSURED_UNDER_18","COUNTY_VIOLENT_CRIME")]),]

# Keep only needed columns
data <- data %>% 
  select(starts_with("TRACT_")|starts_with("COUNTY_")|starts_with("AIR_")|all_of(c("AGE","GENDER","RACETHNICITY","EDUC","MARITAL","EMPLOY","INCOME","REGION9","METRO","HOUSING","HOME_TYPE","SMOKER_CATEGORY","ALCOHOL","PHYSICIANS_TG_DOC","WALK_SCORE")))


# Impute those missing air quality data with median
data$missing.airdata <- factor(ifelse(is.na(data$AIR_QUALITY_MEDIAN_AQI), 1, 0),levels = c(1,0),labels = c("Yes","No"))
data$AIR_QUALITY_GOOD_DAYS <- ifelse(is.na(data$AIR_QUALITY_GOOD_DAYS), median(data$AIR_QUALITY_GOOD_DAYS,na.rm = TRUE), data$AIR_QUALITY_GOOD_DAYS)
data$AIR_QUALITY_MEDIAN_AQI <- ifelse(is.na(data$AIR_QUALITY_MEDIAN_AQI), median(data$AIR_QUALITY_MEDIAN_AQI,na.rm = TRUE), data$AIR_QUALITY_MEDIAN_AQI)
data$AIR_QUALITY_UNHEALTHY_FOR_SENSIT <- ifelse(is.na(data$AIR_QUALITY_UNHEALTHY_FOR_SENSIT), median(data$AIR_QUALITY_UNHEALTHY_FOR_SENSIT,na.rm = TRUE), data$AIR_QUALITY_UNHEALTHY_FOR_SENSIT)
data$AIR_QUALITY_DAYS_PM2_5<- ifelse(is.na(data$AIR_QUALITY_DAYS_PM2_5), median(data$AIR_QUALITY_DAYS_PM2_5,na.rm = TRUE), data$AIR_QUALITY_DAYS_PM2_5)
data$AIR_QUALITY_DAYS_PM10<- ifelse(is.na(data$AIR_QUALITY_DAYS_PM10), median(data$AIR_QUALITY_DAYS_PM10,na.rm = TRUE), data$AIR_QUALITY_DAYS_PM10)
```
