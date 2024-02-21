import os
def outputTime():
    a = os.popen('date').read()
    b = a.split(' ')
    return b[0]+'_' + b[1]+  '_' +b[2]+'_'+b[3] + '_' + b[4]