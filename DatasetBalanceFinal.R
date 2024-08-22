setwd('/Users/maxwellhou/Desktop/PhD/MRes/Team Challenge')
library(RJSONIO)
library(progress)

sCArgScale = 17 #Max arg number

#This one is aimed at an equal number of argmentaiton in all argmented classes

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
  data$argAmount = 0
  data$argType = 'None'
  data$argPar1 = NA
  data$argPar2 = NA
  data$argPerformed = F
  return(data)
}

#Initialise the argmentaiton list
argMethods = c('Blu', 'Rot', 'Flip', 'Noise', 'Bri')
bluList = list(argPar1 = list(c(3, 5, 7)),
               argPar2 = list(head(seq(0, 360, length.out = 5), -1)))
rotList = list(argPar1 = list(tail(head(seq(0, 360, length.out = 5), -1), -1)),
               argPar2 = list(rep(NA, 5)))
flipList = list(argPar1 = list(c('h', 'v')),
                argPar2 = NA)
noiseList  = list(argPar1 = list(c(0.6, 0.7, 0.8)),
                  argPar2 = NA)
briList  = list(argPar1 = list(c(-100, -75, -50, 50, 75, 100)),
                argPar2 = NA)
argList = list(Blu = bluList, Rot = rotList, Flip = flipList, Noise = noiseList, Bri = briList)
argDF = data.frame(row.names = argMethods, lim = c(length(bluList$argPar1[[1]]),
                                                   length(rotList$argPar1[[1]]),
                                                   length(flipList$argPar1[[1]]),
                                                   length(noiseList$argPar1[[1]]),
                                                   length(briList$argPar1[[1]])), used = 0)

#Assign the argmentaiton amout (count)
assArgAmount = function(df, label, times)
{
  if(times > sum(argDF$lim))
  {stop('Are you crazy trying to generate the same image')}
  
  df$choiceCom = as.character(df$choiceCom)
  
  argImages = subset(df, df$choiceCom == label & df$argType == 'None')
  
  df$argAmount[df$X %in% argImages$X] = df$argAmount[df$X %in% argImages$X] + as.integer(times)
  
  totalNum = as.integer((times - as.integer(times)) * nrow(argImages))
  trainNum = as.integer(totalNum * 0.8)
  valNum = totalNum - trainNum
  
  print(totalNum)
  
  trainArgImages = subset(argImages, argImages$split == 'Train')
  valArgImages = subset(argImages, argImages$split == 'Val')
  
  trainArgImages = sample(trainArgImages$X, trainNum, replace = F)
  valArgImages = sample(valArgImages$X, valNum, replace = F)
  
  df$argAmount[df$X %in% trainArgImages] = df$argAmount[df$X %in% trainArgImages] + 1
  df$argAmount[df$X %in% valArgImages] = df$argAmount[df$X %in% valArgImages] + 1
  
  return(df)
}

#Allocate argmentation based on assArgAmout ($argAmout)
assArgAlt = function(df)
{
  argImages = subset(df, df$argAmount > 0 & !df$argPerformed)
  
  pb = progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = sum(argImages$argAmount),    
    width = 60)
  
  for(i in 1:nrow(argImages))
  {
    tempRow = argImages[i,]
    times = tempRow$argAmount
    argNum = length(argMethods)
    tempArgMethods = argMethods
    
    if(times <= 10)
    {
      tempArg = c(rep(argMethods, times %/% argNum), sample(argMethods, times %% argNum, replace = F))
    }
    else if(times > 10 & times <=14)
    {
      tempArg = rep(argMethods, 2)
      tempArgMethods = tempArgMethods[!(tempArgMethods == 'Flip')]
      argNum = length(tempArgMethods)
      tempArg = c(tempArg, rep(tempArgMethods, (times-10) %/% argNum), sample(tempArgMethods, times %% argNum, replace = F))
    }
    else
    {
      tempArg = rep(argMethods, 2)
      tempArgMethods = tempArgMethods[!(tempArgMethods == 'Flip')]
      
      tempArg = c(tempArg, tempArgMethods)
      tempArgMethods = 'Bri'
      
      argNum = length(tempArgMethods)
      tempArg = c(tempArg, rep(tempArgMethods, times - 14))
    }
    
    tempArgList = argList
    
    for(j in tempArg)
    {
      tempRow = argImages[i,]
      
      if(tempRow$split == 'Val')
      {
        tempRow$argType = 'Clone'
        tempRow$argPar1 = NA
        tempRow$argPar2 = NA
        tempRow$argAmount = NA
        pb$tick()
      }
      else
      {
        tempRow$argType = j
        
        #tempArgPar1 = sample(tempArgList[[j]]$argPar1[[1]], 1)
        if(length(tempArgList[[j]]$argPar1[[1]]) > 1)
        {
          tempArgPar1 = sample(tempArgList[[j]]$argPar1[[1]], 1)
        }
        else if(length(tempArgList[[j]]$argPar1[[1]]) == 1)
        {
          tempArgPar1 = tempArgList[[j]]$argPar1[[1]]
        }
        else
        {
          stop('Not enough argPar1 for', j, tempRow)
        }
        tempRow$argPar1 = tempArgPar1
        tempArgList[[j]]$argPar1[[1]] = tempArgList[[j]]$argPar1[[1]][tempArgList[[j]]$argPar1[[1]] != tempArgPar1]
        
        if(length(tempArgList[[j]]$argPar2[[1]]) > 1)
        {
          tempArgPar2 = sample(tempArgList[[j]]$argPar2[[1]], 1)
        }
        else if(length(tempArgList[[j]]$argPar2[[1]]) == 1)
        {
          tempArgPar2 = tempArgList[[j]]$argPar2[[1]]
        }
        else
        {
          stop('Not enough argPar2 for', j, tempRow)
        }
        #tempArgPar2 = sample(tempArgList[[j]]$argPar2[[1]], 1)
        tempRow$argPar2 = tempArgPar2
        tempArgList[[j]]$argPar2[[1]] = tempArgList[[j]]$argPar2[[1]][tempArgList[[j]]$argPar2[[1]] != tempArgPar2]
        
        tempRow$argAmount = NA
        
        pb$tick()
        
      }
      df = rbind(df, tempRow)
    }
  }
  df$argPerformed[df$argAmount > 0] = T
  
  return(df)
}

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
    tempArgMethods = argMethods
    #Masure that flipe is only selected 2 times
    if(times <= 10)
    {
      tempArg = c(rep(argMethods, times %/% argNum), sample(argMethods, times %% argNum, replace = F))
    }
    else if(times > 10 & times <=14)
    {
      tempArg = rep(argMethods, 2)
      tempArgMethods = tempArgMethods[!(tempArgMethods == 'Flip')]
      argNum = length(tempArgMethods)
      tempArg = c(tempArg, rep(tempArgMethods, (times-10) %/% argNum), sample(tempArgMethods, times %% argNum, replace = F))
    }
    else
    {
      tempArg = rep(argMethods, 2)
      tempArgMethods = tempArgMethods[!(tempArgMethods == 'Flip')]
      
      tempArg = c(tempArg, tempArgMethods)
      tempArgMethods = 'Bri'
      
      argNum = length(tempArgMethods)
      tempArg = c(tempArg, rep(tempArgMethods, (times-14) %/% argNum), sample(tempArgMethods, times %% argNum, replace = F))
    }
    
    
    tempArgList = argList
    
    for(j in tempArg)
    {
      tempRow = argImages[i,]
      
      if(tempRow$split == 'Val')
      {
        tempRow$argType = 'Clone'
        tempRow$argPar1 = NA
        tempRow$argPar2 = NA
      }
      else
      {
        tempRow$argType = j
        
        tempArgPar1 = sample(tempArgList[[j]]$argPar1[[1]], 1)
        tempRow$argPar1 = tempArgPar1
        tempArgList[[j]]$argPar1[[1]] = tempArgList[[j]]$argPar1[[1]][tempArgList[[j]]$argPar1[[1]] != tempArgPar1]
        
        tempArgPar2 = sample(tempArgList[[j]]$argPar2[[1]], 1)
        tempRow$argPar2 = tempArgPar2
        tempArgList[[j]]$argPar2[[1]] = tempArgList[[j]]$argPar2[[1]][tempArgList[[j]]$argPar2[[1]] != tempArgPar2]
      }
      
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
    tempRow = argTrainImages[i,]
    
    tempRow$argType = 'Clone'
    tempRow$argPar1 = NA
    tempRow$argPar2 = NA
    
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


#Read the data
#Reorder labels column to make sure that soil dung is converted to dung soil
labelOrder = c('Clover', 'Grass', 'Dung', 'Soil')

data = read.csv('dataset-v6-raw-clean.csv')

data = subset(data, !grepl('Artifact', data$choice))
data = subset(data, !grepl('Sheep Dung', data$choice))
data = subset(data, !data$choice == '')

for(i in 1:dim(data)[1])
{
  if(grepl(':', data$choice[i]))
  {
    tempChoices = fromJSON(data$choice[i])$choice
    tempChoices = factor(tempChoices, levels = labelOrder)
    tempChoices = tempChoices[order(tempChoices)]
    data$choiceCom[i]  = paste(tempChoices, collapse = ', ')
  }
  else
  {
    data$choiceCom[i] = data$choice[i]
  }
}

data$choiceCom = as.factor(data$choiceCom)
dataV8.0 = data

#Check data distribution
countDf = countClass(dataV8.0)
labelCountDf = countLabel(dataV8.0)

#Assign trianing and validation
dataV8.0 = spliteTV(dataV8.0)

#Varify raining and validation splite
piePerClass(dataV8.0)

#Increase special classes 10 times (1+9)
dataV8.1 = argIni(dataV8.0)

countDf = countClass(dataV8.1)
specialClass = subset(countDf$class, countDf$count <= 100)

for(i in specialClass)
{
  print(paste('Processing class', i))
  dataV8.1 = assArgAmount(dataV8.1, i, sCArgScale)
}

dataV8.1 = assArgAlt(dataV8.1)

piePerClass(dataV8.1)
checkSplite(dataV8.1, dataV8.0)

countDf = countClass(dataV8.1)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8)

# #Lines comented below are for increasing the other groups if needed (balance methode 2 and 3)
# #Increase the remaining class by sutible amount to achive same number as clover
# dataV8.2 = dataV8.1
# countDf = countClass(dataV8.2)
# remainingClass = subset(countDf$class, !countDf$class %in% specialClass)
# labelCountDf = countLabel(dataV8.2)
# 

#
# #Dung
# argScale = (labelCountDf['Clover', 'count'] - labelCountDf['Dung', 'count'])/countDf['Grass, Dung', 'count']
# dataV8.2 = assArgAmount(dataV8.2, 'Grass, Dung', argScale)
# dataV8.2 = assArgAlt(dataV8.2)
# 
# dArgScale = argScale
# 
# countDf = countClass(dataV8.2)
# labelCountDf = countLabel(dataV8.2)
# 
# #Soil and grass soil
# argScale = (labelCountDf['Clover', 'count'] - labelCountDf['Soil', 'count'])/sum(countDf[c('Soil', 'Grass, Soil'), 'count'])
# 
# sArScale = argScale
# 
# dataV8.2 = assArgAmount(dataV8.2, 'Soil', argScale)
# dataV8.2 = assArgAmount(dataV8.2, 'Grass, Soil', argScale)
# dataV8.2 = assArgAlt(dataV8.2)
# 
# countDf = countClass(dataV8.2)
# labelCountDf = countLabel(dataV8.2)

dataV8.2 = dataV8.1
countDf = countClass(dataV8.2)
labelCountDf = countLabel(dataV8.2)

#Cut down grass
grassDownSampleNum = min(subset(countDf$count, countDf$class %in% c('Clover', 'Soil', 'Dung')))

gArgScale = grassDownSampleNum/labelCountDf['Grass', 'count']

grassTrainKeepNum = as.integer(grassDownSampleNum * 0.8)
grassValKeepNum = grassDownSampleNum - grassTrainKeepNum

temp = subset(dataV8.2, dataV8.2$choiceCom == 'Grass' & dataV8.2$split == 'Train')
grassTrainKeepRows = temp[sample(nrow(temp), grassTrainKeepNum, replace = F),]
temp = subset(dataV8.2, dataV8.2$choiceCom == 'Grass' & dataV8.2$split == 'Val')
grassValKeepRows = temp[sample(nrow(temp), grassValKeepNum, replace = F),]

dataTemp = subset(dataV8.2, dataV8.2$choiceCom != 'Grass')
dataV8.3 = rbind(dataTemp,
                 grassTrainKeepRows,
                 grassValKeepRows)

countDf = countClass(dataV8.3)
labelCountDf = countLabel(dataV8.3)

#Check
piePerClass(dataV8.3)
checkSplite(dataV8.3, dataV8.0)

#Plot
par(mfrow = c(1,2))
countDf = countClass(dataV8.0)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8)

countDf = countClass(dataV8.3)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8)

#Final preperation
dataV8f = dataV8.3
dataV8f$argImageName = NA
for(i in 1:dim(dataV8f)[1])
{
  if(dataV8f$argType[i] == 'Clone' | dataV8f$argType[i] == 'None')
  {
    dataV8f$argImageName[i] = dataV8f$image[i]
  }
  else
  {
    dataV8f$argImageName[i] = paste(dataV8f[i, c('argType', 'argPar1', 'argPar2')], collapse = '_')
    dataV8f$argImageName[i] = paste(sub("\\.jpg$", "", dataV8f$image[i]), '_', dataV8f$argImageName[i], '.jpg', sep = '')
  }
}

colnames(dataV8f)[1] = ''

#Output
#write.csv(dataV8f, 'dataV6_arg_v8.csv', row.names = F)

#SCRUMBLE
library(mldr)
dataV8fSCR = dataV8f

labelFormatConvert = function(data)
{
  data$Grass = 0
  data$Soil = 0
  data$Dung = 0
  data$Clover = 0
  
  data$choiceCom = as.character(data$choiceCom)
  
  pb = progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = dim(data)[1],    
    width = 60)
  
  for(i in 1:dim(data)[1])
  {
    tempLabel = unlist(strsplit(data$choiceCom[i], ', '))
    data[i, tempLabel] = 1
    pb$tick()
  }
  return(data)
}

par(mfrow = c(1,2))
dataV8.0SCR = labelFormatConvert(dataV8.0)
dataV8.0SCR = dataV8.0SCR[, c('Grass', 'Soil', 'Dung', 'Clover')]
dataV8.0SCR = mldr_from_dataframe(dataV8.0SCR, labelIndices = 1:ncol(dataV8.0SCR))
summary(dataV8.0SCR)
plot(dataV8.0SCR, type = 'LC')

dataV8fSCR = labelFormatConvert(dataV8fSCR)
dataV8fSCR = dataV8fSCR[, c('Grass', 'Soil', 'Dung', 'Clover')]
dataV8fSCR = mldr_from_dataframe(dataV8fSCR, labelIndices = 1:ncol(dataV8fSCR))
summary(dataV8fSCR)
plot(dataV8fSCR, type = 'LC')

print(paste('Special Class Scale:', sCArgScale))
#print(paste('Dung Scale:', dArgScale))
#print(paste('Soil and Grass, Soil Scale:', sArScale))
print(paste('Grass Reduced to Amount:', grassDownSampleNum))
print(paste('SCRUMBLE Imprvment:', dataV8.0SCR$measures$scumble - dataV8fSCR$measures$scumble))


#Check contamination
checkContaimnation = function(df)
{
  df$image = as.factor(df$image)
  
  pb = progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = nlevels(df$image),    
    width = 60)
  
  for(i in levels(df$image))
  {
    imgList = subset(df, df$image == i)
    imgSplitType = subset(imgList$split, imgList$argType == 'None')
    
    if(length(imgSplitType) > 1) {stop('Duplicated source image')}
    
    if(!all(imgList$split == imgSplitType)) {stop(paste('Contamination for image', i))}
    
    pb$tick()
  }
  print('No contamination detected!')
}

checkContaimnation(dataV8f)

#check identical argumentation
argImages = subset(dataV8f, !(dataV8f$argType %in% c('Clone', 'None')))
argImages$argImageName = as.factor(argImages$argImageName)

if(nrow(argImages) == nlevels(argImages$argImageName))
{
  print('No identical argumentation')
}else{
  stop('Identical argumentation detected!')
}

argAllocateAmout = sum(subset(dataV8f$argAmount, dataV8f$split == 'Train'), na.rm = T)

print(paste('Total allocated arg amout:', argAllocateAmout))
print(paste('Total applied arg amout:', nlevels(argImages$argImageName)))

#Check exception arg par
for(i in argMethods)
{
  tempArgPar1List = subset(dataV8f$argPar1, dataV8f$argType == i)
  tempArgPar2List = subset(dataV8f$argPar2, dataV8f$argType == i)
  
  if(!all(tempArgPar1List %in% argList[[i]]$argPar1[[1]] ))
  {
    stop('Blyat, ecption arg')
  }
  else if(!all(tempArgPar2List %in% argList[[i]]$argPar2[[1]] ))
  {
    stop('Blyat, ecption arg')
  }
  else
  {
    print('No argPar expection')
  }
}

#Check if the correct amout of arg is allocated
argAmoutRows = subset(dataV8f, dataV8f$argAmount > 0)
for(i in 1:nrow(argAmoutRows))
{
  tempRows = subset(dataV8f, dataV8f$image == argAmoutRows$image[i])
  tempNRow = nrow(subset(tempRows, tempRows$argType != 'None'))
  
  if(tempNRow != sum(tempRows$argAmount, na.rm = T))
  {stop('Augmentaiton amout is wrong')}
}

#For update
par(mar = c(5.1, 15, 4.1, 2.1))
barplot(countLabel(dataV8.0)$count, names.arg = countLabel(dataV8.0)$label, las = 1, horiz = T)
barplot(countClass(dataV8.0)$count, names.arg = countClass(dataV8.0)$class, las = 1, horiz = T)

barplot(countLabel(dataV8f)$count, names.arg = countLabel(dataV8f)$label, las = 1, horiz = T)
barplot(countClass(dataV8f)$count, names.arg = countClass(dataV8f)$class, las = 1, horiz = T)

data = read.csv('dataV6_arg_v8.csv')
sum(subset(data$argAmount, !(data$argType %in% c('None', 'Clone'))))

#For report
oriolColor = c('#00B1C1', '#D8EDE6', '#3F8976',
               '#005C7E', '#E6F4F1', '#852500',
               '#009E79', '#00614F', '#172C26',
               '#EEE8A9', '#852500', '#3F8976')
barplot(rep(1, 12), names.arg = 1:12, col = oriolColor)

labelCountDf = countLabel(dataV8.0)
classCountDf = countClass(dataV8.0)

par(mfrow = c(1,2))
par(mar = c(5.1, 15, 4.1, 2.1))
par(bg = NA, fg = oriolColor[2], col = oriolColor[2], col.axis = oriolColor[2],
    col.lab = oriolColor[2], col.main = oriolColor[2])

orderedCCDf = classCountDf[order(classCountDf$count),]
barplot(orderedCCDf$count, names.arg = orderedCCDf$class, las = 1, horiz = T,
        main = 'Label Combination Count', col = oriolColor[2])
#barplot(log10(orderedCCDf$count), names.arg = orderedCCDf$class, las = 1, horiz = T)
#write.csv(orderedCCDf, 'ordered data V8.0 combination count.csv', row.names = F)
#par(mar = c(5.1, 4.1, 4.1, 2.1))
labelCountDf = labelCountDf[order(labelCountDf$count),]
barplot(labelCountDf$count, names.arg = labelCountDf$label, las = 1, horiz = T,
        main = 'Total Label Count', col = oriolColor[2])

par(mfrow = c(1,2))
countDf = countClass(dataV8.0)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8, main = 'Before Balancing', radius = 1)

countDf = countClass(dataV8f)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'), cex = 0.8, main = 'After Balancing', radius = 1)

#Plot barplot before and after
baBar = orderedCCDf
baBar$countf = NA
countDf = countClass(dataV8f)

for(i in baBar$class)
{
  baBar[i, 'countf'] = countDf[i, 'count']
}

par(mfrow = c(1,2))
par(mar = c(5.1, 15, 4.1, 2.1))
barplot(baBar$count, names.arg = baBar$class, las = 1, horiz = T,
        main = 'Before Balancing')
barplot(baBar$countf, names.arg = orderedCCDf$class, las = 1, horiz = T,
        main = 'After Balancing')

par(mfrow = c(1,2))
plot(dataV8.0SCR, type = 'LC')
plot(dataV8.0SCR, type = 'LC')

par(mfrow = c(1,2))
par(bg = NA, fg = oriolColor[2], col = oriolColor[2], col.axis = oriolColor[2],
    col.lab = oriolColor[2], col.main = oriolColor[2])
countDf = countClass(dataV8.0)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'),
    cex = 0.8, col = c(oriolColor[4], oriolColor[10]))

countDf = countClass(dataV8f)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'),
    cex = 0.8, col = c(oriolColor[4], oriolColor[10]))

pieChartColor = c('#3eaebf',
                  '#bfb13d',
                  '#bf3d7c',
                  '#33c2be',
                  '#76a650',
                  '#e5566f',
                  '#82E59B',
                  '#359368',
                  '#fd7a61',
                  '#bbf183',
                  '#2f4858',
                  '#ffcd59',
                  '#f9f871')

countDf = countClass(dataV8.0)
countDf$per = countDf$count/sum(countDf$count)
#countDf = countDf[order(countDf$per),]
minorityCountDf = subset(countDf, countDf$count < 100)
tempRow = c(sum(minorityCountDf$count), 'Minority Combinations', sum(minorityCountDf$per))
countDf = subset(countDf, !countDf$class %in% minorityCountDf$class)
countDf = rbind(countDf[1:2,], tempRow, countDf[3:nrow(countDf),])
countDf$count = as.integer(countDf$count)
countDf$per = round(as.numeric(countDf$per), 3) * 100

par(mfrow = c(1,2))
par(bg = NA, fg = oriolColor[2], col = oriolColor[2], col.axis = oriolColor[2],
    col.lab = oriolColor[2], col.main = oriolColor[2])
pie(countDf$count, labels = paste(countDf$class, countDf$per, '%'),
    cex = 0.8, col = c(pieChartColor[1:2], pieChartColor[8:length(pieChartColor)]))

countDf = countClass(dataV8f)
per = round(countDf$count/sum(countDf$count), 3) * 100
pie(countDf$count, labels = paste(countDf$class, per, '%'),
    cex = 0.8, col = pieChartColor)
