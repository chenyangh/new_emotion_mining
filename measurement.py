"""
Written by Nawshad.
This function outputs f-measures given the
predicted preLabels and target label as a list of lists.
See the function call
"""

import numpy as np


def CalculateFM(preLabels, targetLabels):
    # print('PreLabels:', preLabels)
    # print('targetLabels:', targetLabels)

    Pre_Labels = np.transpose(np.array(preLabels))
    Test_Target = np.transpose(np.array(targetLabels))

    # print('Array of Pre_Labels:',Pre_Labels)
    # print('Array of Test_Target:', Test_Target)

    temp = Pre_Labels + Test_Target

    # print('temp:', temp)

    dim = Test_Target.shape
    m = dim[0]
    n = dim[1]

    print('dim:', m, n)

    a = np.zeros(m)
    b = np.zeros(m)
    c = np.zeros(m)
    F_temp = np.zeros(m)

    for i in range(m):
        a[i] = len(np.nonzero(temp[i, :] == 2)[0])
        b[i] = len(np.nonzero(Pre_Labels[i, :] == 1)[0])
        c[i] = len(np.nonzero(Test_Target[i, :] == 1)[0])

        if b[i] + c[i] == 0:
            temp_f = 1
        else:
            temp_f = (2 * a[i]) / (b[i] + c[i])
        # print('F_measure temp_f:', temp_f)
        F_temp[i] = temp_f

    """	
    print('F_temp:',F_temp)
    print('a', a)
    print('b', b)
    print('c', c)
    """

    print("F-measures:")

    Macro_Fmeasure = np.sum(F_temp) / m
    Micro_Fmeasure = (2 * np.sum(a)) / (np.sum(b) + np.sum(c))

    """
    print('Macro FM:', Macro_Fmeasure)
    print('Micro FM:', Micro_Fmeasure)
    """
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    F_temp = np.zeros(n)

    for i in range(n):
        a[i] = len(np.nonzero(temp[:, i] == 2)[0])
        b[i] = len(np.nonzero(Pre_Labels[:, i] == 1)[0])
        c[i] = len(np.nonzero(Test_Target[:, i] == 1)[0])

        if b[i] + c[i] == 0:
            temp_f = 1
        else:
            temp_f = (2 * a[i]) / (b[i] + c[i])
        F_temp[i] = temp_f
    Exam_Fmeasure = sum(F_temp) / n
    # print('Exam_Fmeasure:', Exam_Fmeasure)

    '''
    for ftemp in F_temp:
        print(ftemp)
    '''

    return {'MacroFM': Macro_Fmeasure, 'MicroFM': Micro_Fmeasure, 'ExamFM': Exam_Fmeasure}


"""
preLabels = 	[[-1,-1,1,-1,-1,1,-1,-1,-1],
				 [-1,-1,-1,1,1,-1,-1,-1,-1],
				 [-1,-1,-1,1,-1,1,-1,-1,-1]]

targetLabels = [[-1,-1,1,1,-1,-1,-1,-1,-1],
				[1,-1,-1,-1,1,-1,-1,-1,-1],
				[-1,-1,1,1,-1,-1,-1,-1,-1]]

results = CalculateFM(preLabels, targetLabels)

print('Results:', results)
"""

