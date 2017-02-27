
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


bookname = "./plots/sentimentdata/Sentimentdata.csv"
book = xlrd.open_workbook(bookname)


posvgood = []
negvgood = []
neuvgood = []
posvbad = []
negvbad = []
neuvbad = []

#--------------baad-----------------
def equalsizedata(data, summ):
    if len(data) < len(summ):
        i = 0 
        j = len(data)
        while i < (len(summ)-j):
            data = np.append(data, 0)
            i = i + 1
    else:
        i = 0
        j = len(summ)
        while i < (len(data)-j):
            summ = np.append(summ, 0)
            i = i + 1
    return data, summ
#-----------------------------

def setrange100(val):
    ip = np.linspace(0,len(val),num = 100)
    xval = np.array([])
    for i in range(1,len(val)):
        xval = np.append(xval, i)
        xval = map(float, xval)
    xval = np.append(xval, i+1)
    fine = np.interp(ip,xval,val)
    return fine


#-----------------------------
def readsetval(val, sheetnm, colno):
    val = np.append(val,sheetnm.col_values(colno))
    val[0] = 0
    val = map(float, val)
    val = setrange100(val)
    return val

#---------------------baad--------

def sumofcols(data, summ, colno):
    '''data = np.array([])
    data = np.append(data, sheet.col_values(colno))
    data[0] = 0
    summ = np.array([])
    summ = np.append(summ,sheetnm.col_values(colno))
    summ[0] = 0
    '''

    #sentence numbers may not be equal
    if len(data) != len(summ):
        data, summ = setrange100(data, summ)
    
    summ = np.add(summ, data)
    return summ


#-------------------------------

def produceplotval(val):
    val = np.convolve(val,[0.2,0.2,0.2,0.2,0.2])
    ip = np.linspace(1,len(val),100)
    xval = np.array([])
    for i in range(1,len(val)):
        xval = np.append(xval, i)
        xval = map(float, xval)
    j=0
    ii = i
    while j< (len(val)-ii): 
        i = i+1
        xval = np.append(xval,i)
        j = j + 1
    fine = np.interp(ip,xval,val)
    
    return fine


#-------------------------------

def plotcurvedata(good,bad, title):
    plt.ylabel("emotion range")
    plt.xlabel("sentence wise propagation")
    plt.title(title)
    plt.plot(good, 'b', bad, 'r')
    figname = './plots/sentimentdata/'+title + ".png"
    plt.savefig(figname)
    plt.clf()

#-----------------------------
def sumselfcol(val):
    summ = float(0.0)
    for i in range(0,len(val)):
        summ = np.add(summ, val[i])
    return summ

#-----------------------------

def plotalldata(posvgood, posvbad, negvgood, negvbad, neuvgood, neuvbad, title):
    plt.ylabel("emotion range")
    plt.xlabel("sentence wise propagation (pos+, neg*, neu.)")
    plt.title(title)
    plt.plot(posvgood, 'b+', posvbad, 'r+', negvgood, 'b*', negvbad, 'r*', neuvgood, 'b.', neuvbad, 'r.')
    figname = './plots/sentimentdata/'+title + "Curve.png"
    plt.savefig(figname)
    plt.clf()

#-----------------------------

def plotbardata(posvgood, posvbad, negvgood, negvbad, neuvgood, neuvbad, title):
    plt.ylabel("emotion range")
    plt.xlabel("Sum of avg emotion in talk")
    plt.title(title)
    objects = ('posgood', 'posbad', 'neggood', 'negbad', 'neugood', 'neubad')
    y_pos = np.arange(len(objects))
    y_val = [posvgood, posvbad, negvgood, negvbad, neuvgood, neuvbad]
    plt.bar(y_pos, y_val, align = 'center', alpha = 0.5)
    plt.xticks(y_pos, objects)
    figname = './plots/sentimentdata/'+title + "Bar.png"
    plt.savefig(figname)
    plt.clf()

#---------------------------

#good talks
print "\n\ngood data-------------"
sheetname = 'talk1'
sheetnm = book.sheet_by_name(sheetname)
negvgood = readsetval(negvgood, sheetnm, 2)
neuvgood = readsetval(neuvgood, sheetnm, 3)
posvgood = readsetval(posvgood, sheetnm, 4)


for sheet in book.sheets():
    #skipping sime sheets
    if sheet.name in ['talk1','talkinfo']:
        continue    

    print sheet.name

    #neg
    data = np.array([])

    data = readsetval(data, sheet, 2)
    negvgood = np.add(negvgood, data)
    

    #neu
    data = np.array([])
    data = readsetval(data, sheet, 3)
    neuvgood = np.add(neuvgood, data)

    #pos
    data = np.array([])
    data = readsetval(data, sheet, 4)
    posvgood = np.add(posvgood, data)

    #20 top talks
    if sheet.name in ['talk20']:
        break

negvgood = np.divide(negvgood, 20)
neuvgood =  np.divide(neuvgood, 20)
posvgood = np.divide(posvgood, 20)


#bad talks
print "\n\nbad data---------------"
sheetname = 'talk_1'
sheetnm = book.sheet_by_name(sheetname)
negvbad = readsetval(negvbad, sheetnm, 2)
neuvbad = readsetval(neuvbad, sheetnm, 3)
posvbad = readsetval(posvbad, sheetnm, 4)

for sheet in book.sheets():
    #skipping sime sheets
    if sheet.name in ['talk_1','talk1','talk2','talk4','talk3','talk5','talk6','talk7',
    'talk8','talk9','talk10','talk11','talk12','talk13','talk14','talk15',
    'talk16','talk17','talk18','talk19','talk20','talkinfo']:
        continue    

    print sheet.name

    #neg
    data = np.array([])
    data = readsetval(data, sheet, 2)
    negvbad = np.add(negvbad, data)
    

    #neu
    data = np.array([])
    data = readsetval(data, sheet, 3)
    neuvbad = np.add(neuvbad, data)

    #pos
    data = np.array([])
    data = readsetval(data, sheet, 4)
    posvbad = np.add(posvbad, data)

    #20 last talks
    if sheet.name in ['talk_20']:
        break


#average over 20
negvbad = np.divide(negvbad, 20)
neuvbad =  np.divide(neuvbad, 20)
posvbad = np.divide(posvbad, 20)

#smoothing
# Smoothing the signal after averaging the data is BAD
negvgood = produceplotval(negvgood)
neuvgood = produceplotval(neuvgood)
posvgood = produceplotval(posvgood)
negvbad = produceplotval(negvbad)
neuvbad = produceplotval(neuvbad)
posvbad = produceplotval(posvbad)


#individual curve plt
plotcurvedata(posvgood, posvbad, "Positive Sentence Progatation (avg over 20) for Good VS Bad Talk")
plotcurvedata(negvgood, negvbad, "Negative Sentence Progatation (avg over 20) for Good VS Bad Talk")
plotcurvedata(neuvgood, neuvbad, "Neutral Sentence Progatation (avg over 20) for Good VS Bad Talk")

#combined curve plt
plotalldata(posvgood, posvbad, negvgood, negvbad, neuvgood, neuvbad, "Sentiment (avg over 20) for Good VS Bad")

#sum of sentiment in whole talk
posvgood= sumselfcol(posvgood)
posvbad = sumselfcol(posvbad)
negvgood= sumselfcol(negvgood)
negvbad= sumselfcol(negvbad)
neuvgood= sumselfcol(neuvgood)
neuvbad = sumselfcol(neuvbad)
print posvgood, posvbad, negvgood, negvbad, neuvgood, neuvbad
plotbardata(posvgood, posvbad, negvgood, negvbad, neuvgood, neuvbad, "Sentiment (total of avg over 20) for Good VS Bad")



