import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import pandas as pd
import numpy as np
import my_functions as mf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random

def include_tree(row):
    x2 = row[1]
    x6 = row[5]
    x7 = row[6]
    x8 = row[7]
    x13 = row[12]

    if x2 <= 6.85:
        if x7 <= 0.9846:
            if x6 <= 1.0045:
                tt = 'A'
            elif x6 > 1.0045:
                tt = 'B'
        elif x7 > 0.9846:
            tt = 'C'
    elif x2 > 6.85:
        if x13 <= 9.45:
            tt = 'D'
        elif x13 > 9.45:
            tt = 'E'
    return tt


def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp

def my_pred_fun(beta_0,beta,temp_x_trigger):
    temp = pd.DataFrame()
    for i in range(temp_x_trigger.shape[0]):
        exp_temp = beta_0 + np.inner(beta,temp_x_trigger.iloc[i,:].tolist())
        temp1 = np.exp(exp_temp)/(1+np.exp(exp_temp))
        temp = temp.append([temp1])
    temp = list(temp.iloc[:,0])
    b=list(np.zeros(60))
    b.extend(temp)
    b = pd.DataFrame(b)
    return b


class App:
    def __init__(self, master):

        os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')
        temp = pd.read_csv('data_dump_24_3_17.csv')
        print(temp.shape)
        temp = temp.dropna()
        print(temp.shape)

        X_all_left = pd.DataFrame(temp.iloc[:,0:20])
        X_all_trigger = pd.DataFrame(temp.iloc[:,20:40])
        X_all_right = pd.DataFrame(temp.iloc[:,40:60])
        Y_all_clean = temp.iloc[:,60]
        index_file_clean = temp.iloc[:,61]

        X_all_trigger.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20"]
        X_all_left.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20"]
        X_all_right.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20"]

        print(X_all_trigger.shape,' : Size of all X trigger')
        print(X_all_left.shape,' : Size of all X left')
        print(X_all_right.shape,' : Size of all X right')
        print(Y_all_clean.shape,' : Size of all Y')
        print(index_file_clean.shape,' : Shape of index file')

        temp = X_all_trigger.apply(include_tree,axis=1)

        data_logit = pd.concat([X_all_trigger,temp],axis=1)

        print(X_all_trigger.shape)
        print(data_logit.shape)
        
        global woe_table
        woe_table = mf.calc_WOE(data_logit.iloc[:,-1],Y_all_clean)

        temp2 = data_logit.apply(subs_WOE,axis=1)

        # just to make sure everything is going okay
        temp3 = pd.concat([temp2,data_logit.iloc[:,-1]],axis=1)

        data_logit = pd.concat([X_all_trigger,temp2],axis=1)

        write_data = pd.concat([data_logit,Y_all_clean],axis=1)
        print(write_data.shape)        
        write_data.to_csv('logit_data_6_4.csv',index=False)


        clf_logit_trigger = LogisticRegression()
        clf_logit_trigger =clf_logit_trigger.fit(data_logit,Y_all_clean)
        beta_0 = clf_logit_trigger.intercept_
        global beta
        beta = clf_logit_trigger.coef_[0].tolist()
        beta_0_back = clf_logit_trigger.intercept_
        global beta_back
        beta_back = clf_logit_trigger.coef_[0].tolist()
        
        print('Initial Values')
        print(beta_0,beta)


        file_name = "D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/breakout files/d5122018.csv"
        title_name = file_name.split('/')[-1]
        file = pd.read_csv(file_name)
        file = file.iloc[2500:3000,:].reset_index()

        TC_layer = 12
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)
        tt3 = 'TC' + str(TC_layer + 40)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        L3 = file.loc[:,tt3]
        ML = file.loc[:,'M.level']
        CS = file.loc[:,'C.speed']
        CP = file.loc[:,'C.percent']
        MW = file.loc[:,'M.width']

        TC_layer_left = mf.find_left(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_left)
        tt2 = 'TC' + str(TC_layer_left + 20)

        LL1 = file.loc[:,tt1]
        LL2 = file.loc[:,tt2]

        TC_layer_right = mf.find_right(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_right)
        tt2 = 'TC' + str(TC_layer_right + 20)

        LR1 = file.loc[:,tt1]
        LR2 = file.loc[:,tt2]

        TC_layer_opp = mf.find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        tt3 = 'TC' + str(TC_layer_opp + 40)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]
        LO3 = file.loc[:,tt3]

        # making the continuous x's
        temp_x_trigger = mf.make_cont_x(L1,L2,ML,CP,CS,LO1,LO2,MW)
        # including the tree based information
        temp = temp_x_trigger.apply(include_tree,axis=1)
        temp_data = pd.concat([temp_x_trigger,temp],axis=1)
        temp2 = temp_data.apply(subs_WOE,axis=1)
        temp_x_trigger = pd.concat([temp_x_trigger,temp2],axis=1)
        logit_trigger = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger))
        logit_trigger = list(logit_trigger.iloc[:,-1])
        b = list(np.zeros(60))
        b.extend(logit_trigger)
        b = pd.DataFrame(b)
        # Create a container
        frame = tkinter.Frame(master)

        var = tkinter.StringVar(frame)
        var.set("Which feature do you want") # initial value
        choices = ['1 : slope_n(l2,3)','2 : slope_n(l2,5)','3 : sign_present(l1,l2,2)','4 : sign_present(l1,l2,5)','5 : sign_present(l1,l2,7)','6 : last_nm(l1,5,32)','7 : last_nm(l2,5,32)','8 : np.mean(cp)','9 : np.std(l1.iloc[-12:])','10 : np.std(ml.iloc[-12:])','11 : find_l1_peak_slope(l1,lo1,cs)','12 : find_l1_peak_ratio(l1,lo1,cs)','13 : find_drop(l1)','14 : find_drop(l2)','15 : cs_change(cs)','16 : get_kinks(l2)','17 : first_derivative(l2)','18 : second_derivative(l2)','19 : bin_crossover(l1,l2)','20 : peak_diff(l1,l2)']
        option = tkinter.OptionMenu(frame,var,*choices)
        option.pack(side = tkinter.LEFT)

        slider = tkinter.Scale(frame, from_=-10, to=10, resolution=0.01, orient=tkinter.HORIZONTAL,length = 750)
        slider.pack(side = tkinter.LEFT)
        
        self.button_select = tkinter.Button(frame,text="Select the value ?",
                                        command=lambda:self.select(var.get(),slider.get(),beta_0,beta,temp_x_trigger,choices))
        self.button_select.pack(side = tkinter.LEFT)

        self.button_reset = tkinter.Button(frame,text = "Reset",command=lambda:self.reset_func(clf_logit_trigger,beta_0,temp_x_trigger)).pack(side = tkinter.LEFT)
        self.button_print_beta_all= tkinter.Button(frame,text="Print All Beta",command=self.print_beta_all).pack(side = tkinter.LEFT)
        self.button_print_beta_selected= tkinter.Button(frame,text="Print Selected Beta",command=lambda:self.print_beta_selected(choices,var.get())).pack(side = tkinter.LEFT)

##        self.textBox = tkinter.Text(frame,height=10, width = 25)
##        self.textBox.insert(tkinter.END,beta)
##        self.textBox.config(state=tkinter.DISABLED)
##        self.textBox.pack(side = tkinter.BOTTOM)

        fig = Figure()
        fig.suptitle(title_name)
        ax = fig.add_subplot(311)
        ax.set_ylim(0,1)
        self.line, = ax.plot(range(len(b)),b)
        ax.grid(b=True, which='both')
        
        ax2 = fig.add_subplot(312)
        ax2.plot(range(len(L1)),L1,range(len(L2)),L2)
        ax2.legend(["L1","L2"])
        ax2.grid(b=True, which='both')

        ax3 = fig.add_subplot(313)
        ax3.plot(range(len(ML)),ML,range(len(CS)),100*CS)
        ax3.legend(["Mold Level","100x Casting Speed"])
        ax3.grid(b=True, which='both')
        
        self.canvas = FigureCanvasTkAgg(fig,master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

    def select(self,beta_name,beta_value,beta_0,beta,temp_x_trigger,choices):
        index_val = choices.index(beta_name)
        beta[index_val]= float(beta_value)
        yList = my_pred_fun(beta_0,beta,temp_x_trigger)
        print("length of the y list",len(yList))
        self.line.set_ydata(yList)
        self.canvas.draw()

    def reset_func(self,clf_logit_trigger,beta_0,temp_x_trigger):
        global beta
        print('Called reset_func')
        beta = clf_logit_trigger.coef_[0].tolist()
        yList = my_pred_fun(beta_0,beta,temp_x_trigger)
        print("length of the y list",len(yList))
        self.line.set_ydata(yList)
        self.canvas.draw()

    def print_beta_all(self):
        print('Called print_beta_all')
        print(beta)
        print(beta_back)

    def print_beta_selected(self,choices,beta_name):
        global beta_back
        print('Called print_beta_selected')
        index_val = choices.index(beta_name)
        print(beta_name,beta_back[index_val])

root = tkinter.Tk()
app = App(root)
root.mainloop()
