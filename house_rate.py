import numpy as np
import copy ,math
#flow chart (input and output)
y_axis=np.array([100,200,300,400])
# y_axis=np.array([300,500])
x_axis=np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
w=np.zeros(x_axis.shape[0])
# x_axis=np.array([1,2])
def func(x,w,b):
    n=np.dot(w,x)+b
    return n
#cost function for the program
def func_cost(w,b,x,y):
    l=x.shape[0]
    m=np.zeros(l)
    cost_sum=0
    for i in range(l):
        n=np.dot(w,x[i])+b
        m=(n-y[i])**2
        cost_sum=cost_sum+m
    total_sum=cost_sum/(2*l)
    return total_sum
#gradient desent
def gradient(x,b,w,y):
    m,l=x.shape
    jw=np.zeros(l)
    jb=0
    for i in range(m):
        err=(np.dot(w,x[i])+b)-y[i]
        for j in range(l):
            jw=jw+(err*x[i,j])/m
            jb=jb+(err)/m
    return jw,jb
# final gradient desent function
def gradient_desent(w_in,b_in,x,y,nums,alpha):
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(nums):
        jw,jb=gradient(x,b,w,y)
        b=b-alpha*jb
        w=w-alpha*jw
    return w,b
print(gradient_desent(w,0,x_axis,y_axis,100,1.0e-2))
print(func_cost([24.64376465, 24.64376465, 24.64376465, 24.64376465],4.251720410845372,x_axis,y_axis))
print(func([6,6,6,6],[24.64376465, 24.64376465, 24.64376465, 24.64376465],4.251720410845372))