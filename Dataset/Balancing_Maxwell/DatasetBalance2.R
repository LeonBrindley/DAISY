setwd('/Users/maxwellhou/Desktop/PhD/MRes/Team Challenge')
library(RJSONIO)
library(progress)
#Conduct datasetV5.1 production following the below requirments
#Deliverble 5 different data set with differnet balancing methode with SCRUMBLE matrix from mldr
#Each dataset should have cloned and argmented images
#tain and validation set should be splite before balance

#define universal functions

#Assign 80:20 training and validation splite
#Note that the as.integer means 1 and even cases are in the validation set, meansing that real distribution is <80:>20
spliteTV = function(data)
{
  data$split = 'a'
  for(i in levels(data$choiceCom))
  {
    tempIndex = which(data$choiceCom == i)
    tempValNum = as.integer(length(tempIndex) * 0.2)
    if(tempValNum < 1 && length(tempIndex > 1))
    {
      data$split[sample(tempIndex, 1, replace = F)] = 'Val' #This is to ensure if number 1, it is in validation, if number * 0.2 < 1, at least one is in trianing
    }
    else{data$split[sample(tempIndex, as.integer(length(tempIndex) * 0.2), replace = F)] = 'Val'}
  }
  data$split[data$split != 'Val'] = 'Train'
  return(data)
}

#Plot splite per per class
piePerClass = function(data)
{
  par(mfrow = c(4,4))
  for(i in levels(as.factor(data$choiceCom)))
  {
    temp = as.factor(subset(data$split, data$choiceCom == i))
    per = round(summary(temp)/length(temp), 5) * 100
    pie(summary(temp),
        main = paste(i, 'Size', length(temp)), labels = paste(levels(temp), per, '%'))
  }
  temp = as.factor(data$split)
  per = round(summary(temp)/length(temp), 5) * 100
  pie(summary(temp),
      main = paste('Total', 'Size', length(temp)), labels = paste(levels(temp), per, '%'))
  par(mfrow = c(1,1))
}

#Check if splite ratiaos are excatly the same
checkSplite = function(newData, oriData)
{
  newData$choiceCom = as.factor(newData$choiceCom)
  oriData$choiceCom = as.factor(oriData$choiceCom)
  newData$split = as.factor(newData$split)
  oriData$split = as.factor(oriData$split)
  spliteDf = data.frame(newDataVal = rep(NA, length(levels(newData$choiceCom))),
                        newDataTrain = rep(NA, length(levels(newData$choiceCom))),
                        newDataValRatio = rep(NA, length(levels(newData$choiceCom))),
                        oriDataVal = rep(NA, length(levels(oriData$choiceCom))),
                        oriDataTrain = rep(NA, length(levels(oriData$choiceCom))),
                        oriDataValRatio = rep(NA, length(levels(oriData$choiceCom))),
                        row.names = levels(newData$choiceCom))
  
  for(i in rownames(spliteDf))
  {
    spliteDf[i, c('newDataTrain', 'newDataVal')] = summary(subset(newData$split, newData$choiceCom == i))
    spliteDf[i, c('oriDataTrain', 'oriDataVal')] = summary(subset(oriData$split, oriData$choiceCom == i))
  }
  spliteDf$newDataValRatio = spliteDf$newDataVal/(spliteDf$newDataTrain + spliteDf$newDataVal)
  spliteDf$oriDataValRatio = spliteDf$oriDataVal/(spliteDf$oriDataTrain + spliteDf$oriDataVal)
  spliteDf$identical = spliteDf$newDataValRatio == spliteDf$oriDataValRatio
  spliteDf$valArgRatio = spliteDf$newDataVal/spliteDf$oriDataVal
  spliteDf$trainArgRatio = spliteDf$newDataTrain/spliteDf$oriDataTrain
  
  print(paste('Two data set are', identical(spliteDf$newDataValRatio, spliteDf$oriDataValRatio), 'identical'))
  return(spliteDf)
}

#Initialise df for argmentaiton
argIni = function(data)
{
  data$argType = 'None'
  data$argPar1 = NA
  data$argPar2 = NA
  return(data)
}

#Initialise the argmentaiton list
argMethods = c('Blu', 'Rot', 'Flip', 'Noise', 'Bri')
bluList = list(argPar1 = list(seq(5, 10, length.out = 5)),
               argPar2 = list(head(seq(0, 360, length.out = 5), -1)))
rotList = list(argPar1 = list(tail(head(seq(0, 360, length.out = 5), -1), -1)),
               argPar2 = list(rep(NA, 5)))
flipList = list(argPar1 = list(c('h', 'v')),
                argPar2 = NA)
noiseList  = list(argPar1 = list(seq(5, 10, length.out = 5)),
                  argPar2 = NA)
briList  = list(argPar1 = list(seq(5, 10, length.out = 5)),
                argPar2 = NA)
argList = list(Blu = bluList, Rot = rotList, Flip = flipList, Noise = noiseList, Bri = briList)

#this function increase the label by a given integar times
assArgInt = function(df, label, times)
{
  df$choiceCom = as.character(df$choiceCom)
  argNum = length(argMethods)
  
  argImages = subset(df, df$choiceCom == label & df$argType == 'None')
  
  pb = progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = dim(argImages)[1] * times,    
    width = 60)
  
  print(dim(argImages)[1] * times)
  
  for(i in 1:dim(argImages)[1])
  {
    tempArg = c(rep(argMethods, times %/% argNum), sample(argMethods, times %% argNum, replace = F))
    for(j in tempArg)
    {
      tempRow = argImages[i,]
      tempArgList = argList
      
      tempRow$argType = j
      
      tempArgPar1 = sample(tempArgList[[j]]$argPar1[[1]], 1)
      tempRow$argPar1 = tempArgPar1
      tempArgList[[j]]$argPar1[[1]] = tempArgList[[j]]$argPar1[[1]][tempArgList[[j]]$argPar1[[1]] != tempArgPar1]
      
      tempArgPar2 = sample(tempArgList[[j]]$argPar2[[1]], 1)
      tempRow$argPar2 = tempArgPar2
      tempArgList[[j]]$argPar2[[1]] = tempArgList[[j]]$argPar2[[1]][tempArgList[[j]]$argPar2[[1]] != tempArgPar2]
      
      df = rbind(df, tempRow)
      
      pb$tick()
    }
  }
  
  return(df)
}

#this number increase by one time of a subset of the label with number of images
assArgDou = function(df, label, number)
{
  df$choiceCom = as.character(df$choiceCom)
  argImages = subset(df, df$choiceCom == label & df$argType == 'None')
  
  argTrainNumber = as.integer(number * 0.8) #Train set is slightly smaller, also there is always one in validation set
  argValNumber = number - argTrainNumber
  
  argValImages = subset(argImages, argImages$split == 'Val')
  argValImages = argValImages[sample(nrow(argValImages), argValNumber, replace = F),]
  
  argTrainImages = subset(argImages, argImages$split == 'Train')
  argTrainImages = argTrainImages[sample(nrow(argTrainImages), argTrainNumber, replace = F),]
  
  pb = progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = number,    
    width = 60)   
  
  for(i in 1:dim(argValImages)[1])
  {
    tempArg = sample(argMethods, 1, replace = F)
    tempRow = argValImages[i,]
    tempArgList = argList
    
    tempRow$argType = tempArg
    j = tempArg
    
    tempArgPar1 = sample(tempArgList[[j]]$argPar1[[1]], 1)
    tempRow$argPar1 = tempArgPar1
    tempArgList[[j]]$argPar1[[1]] = tempArgList[[j]]$argPar1[[1]][tempArgList[[j]]$argPar1[[1]] != tempArgPar1]
    
    tempArgPar2 = sample(tempArgList[[j]]$argPar2[[1]], 1)
    tempRow$argPar2 = tempArgPar2
    tempArgList[[j]]$argPar2[[1]] = tempArgList[[j]]$argPar2[[1]][tempArgList[[j]]$argPar2[[1]] != tempArgPar2]
    
    df = rbind(df, tempRow)
    pb$tick()
  }
  
  for(i in 1:dim(argTrainImages)[1])
  {
    tempArg = sample(argMethods, 1, replace = F)
    tempRow = argTrainImages[i,]
    tempArgList = argList
    
    tempRow$argType = tempArg
    j = tempArg
    
    tempArgPar1 = sample(tempArgList[[j]]$argPar1[[1]], 1)
    tempRow$argPar1 = tempArgPar1
    tempArgList[[j]]$argPar1[[1]] = tempArgList[[j]]$argPar1[[1]][tempArgList[[j]]$argPar1[[1]] != tempArgPar1]
    
    tempArgPar2 = sample(tempArgList[[j]]$argPar2[[1]], 1)
    tempRow$argPar2 = tempArgPar2
    tempArgList[[j]]$argPar2[[1]] = tempArgList[[j]]$argPar2[[1]][tempArgList[[j]]$argPar2[[1]] != tempArgPar2]
    
    df = rbind(df, tempRow)
    pb$tick()
  }
  
  return(df)
}

#Get class count df
countClass = function(data)
{
  temp = as.data.frame(summary(as.factor(data$choiceCom)))
  colnames(temp) = 'count'
  temp$class = rownames(temp)
  return(temp)
}

countLabel = function(data)
{
  label = c('Grass', 'Clover', 'Soil', 'Dung')
  tempDf = data.frame(label, count = NA)
  rownames(tempDf) = tempDf$label
  
  for(i in tempDf$label)
  {
    tempDf[i, 'count'] = length(subset(data$choiceCom,grepl(i, data$choiceCom)))
  }
  
  return(tempDf)
}


#Re-read the data
data = read.csv('dataset-v5-raw-clean.csv')
for(i in 1:dim(data)[1])
{
  if(grepl(':', data$choice[i]))
  {
    data$choiceCom[i] = paste(fromJSON(data$choice[i])$choice, collapse = ', ')
  }
  else
  {
    data$choiceCom[i] = data$choice[i]
  }
}

data = subset(data, !grepl('Artifact', data$choiceCom))
data = subset(data, !grepl('Sheep Dung', data$choiceCom))
data = subset(data, !data$choiceCom == '')
data$choiceCom = as.factor(data$choiceCom)
dataV5.0 = data

#Assign trianing and validation
dataV5.0 = spliteTV(dataV5.0)

#Varify raining and validation splite
piePerClass(dataV5.0)

#Increase special classes 10 times (1+9)
dataV5.1 = argIni(dataV5.0)

countDf = countClass(dataV5.1)
specialClass = subset(countDf$class, countDf$count < 100)

for(i in specialClass)
{
  print(paste('Processing class', i))
  dataV5.1 = assArgInt(dataV5.1, i, 9)
}

piePerClass(dataV5.1)
checkSplite(dataV5.1, dataV5.0)

#Increase the remaining class by sutible amount to achive same number as clover
dataV5.2 = dataV5.1
countDf = countClass(dataV5.2)
remainingClass = subset(countDf$class, !countDf$class %in% specialClass)
labelCountDf = countLabel(dataV5.2)

#Dung
argScale = (labelCountDf['Clover', 'count'] - labelCountDf['Dung', 'count'])/countDf['Grass, Dung', 'count']
dataV5.2 = assArgInt(dataV5.2, 'Grass, Dung', as.integer(argScale))

countDf = countClass(dataV5.2)
labelCountDf = countLabel(dataV5.2)
argScale = labelCountDf['Clover', 'count'] - labelCountDf['Dung', 'count']
dataV5.2 = assArgDou(dataV5.2, 'Grass, Dung', argScale)

#Soil and grass soil
argScale = (labelCountDf['Clover', 'count'] - labelCountDf['Soil', 'count'])/sum(countDf[c('Soil', 'Grass, Soil'), 'count'])
dataV5.2 = assArgInt(dataV5.2, 'Soil', as.integer(argScale))
dataV5.2 = assArgInt(dataV5.2, 'Grass, Soil', as.integer(argScale))

countDf = countClass(dataV5.2)
labelCountDf = countLabel(dataV5.2)

argScale = labelCountDf['Clover', 'count'] - labelCountDf['Soil', 'count']

#Splite accrording to current count ratio for soil and grass soil
soilArgScale = as.integer(argScale * (countDf['Soil', 'count']/countDf['Grass, Soil', 'count']))
grassSoilArgScale = argScale - soilArgScale

dataV5.2 = assArgDou(dataV5.2, 'Soil', soilArgScale)
dataV5.2 = assArgDou(dataV5.2, 'Grass, Soil', grassSoilArgScale)

countDf = countClass(dataV5.2)
labelCountDf = countLabel(dataV5.2)

#Cut down grass
grassDownSampleNum = min(subset(countDf$count, countDf$class %in% c('Clover', 'Soil', 'Dung')))

grassTrainKeepNum = as.integer(grassDownSampleNum * 0.8)
grassValKeepNum = grassDownSampleNum - grassTrainKeepNum

temp = subset(dataV5.2, dataV5.2$choiceCom == 'Grass' & dataV5.2$split == 'Train')
grassTrainKeepRows = temp[sample(nrow(temp), grassTrainKeepNum, replace = F),]
temp = subset(dataV5.2, dataV5.2$choiceCom == 'Grass' & dataV5.2$split == 'Val')
grassValKeepRows = temp[sample(nrow(temp), grassValKeepNum, replace = F),]

dataTemp = subset(dataV5.2, dataV5.2$choiceCom != 'Grass')
dataV5.3 = rbind(dataTemp,
                 grassTrainKeepRows,
                 grassValKeepRows)

countDf = countClass(dataV5.3)
labelCountDf = countLabel(dataV5.3)

#Check
piePerClass(dataV5.3)
checkSplite(dataV5.3, dataV5.0)

#Plot
par(mfrow = c(1,2))
countDf = countClass(dataV5.0)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8)

countDf = countClass(dataV5.3)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8)

#Final preperation
dataV5f = dataV5.3
dataV5f$argImageName = apply(dataV5f[, c('argType', 'argPar1', 'argPar2')], 1, function(x) paste(x, collapse = "_"))
dataV5f$argImageName = paste(sub("\\.jpg$", "", dataV5f$image), '_', dataV5f$argImageName, '.jpg', sep = '')
colnames(dataV5f)[1] = ''

#Output
write.csv(dataV5f, 'dataV5_arg_v1.csv', row.names = F)





