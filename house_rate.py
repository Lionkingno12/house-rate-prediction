import numpy as np
import copy ,math
#flow chart (input and output)
y_axis=np.array([271.5,232,509.8])
# y_axis=np.array([300,500])
x_axis=np.array([[952,2,1,65],[1244,3,2,64],[1947,3,2,17]])
w=np.zeros(x_axis.shape[1])
# x_axis=np.array([1,2])
def z_normalization(x):
    mu=np.mean(x,axis=0)
    sigma=np.mean(x,axis=0)
    x_normal=(x-mu)/sigma
    return x_normal
def func(x,w,b):
    n=np.dot(w,x)+b
    return n
#cost function for the program
def func_cost(w,b,x,y):
    x= z_normalization(x)
    l=x.shape[0]
    cost_sum=0
    for i in range(l):
        n=np.dot(w,x[i])+b
        sq_erro=(n-y[i])**2
        cost_sum=cost_sum+sq_erro
    total_sum=cost_sum/(2*l)
    return total_sum
#gradient desent
def gradient(x,b,w,y):
    x=z_normalization(x)
    m,l=x.shape
    jw=np.zeros(l)
    jb=0
    for i in range(m):
        err=(np.dot(w,x[i])+b)-y[i]
        for j in range(l):
            jw[j]=jw[j]+(err*x[i,j])/m
        jb=jb+(err)/m
    return jw,jb
# final gradient desent function
def gradient_desent(w_in,b_in,x,y,nums,alpha):
    x=z_normalization(x)
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(nums):
        jw,jb=gradient(x,b,w,y)
        b=b-alpha*jb
        w=w-alpha*jw
    return w,b
print(gradient_desent(w,0,x_axis,y_axis,100000,1e-7))
print(func_cost([ 0.26101297, -0.12314478, -0.14273909, -0.50344429],0.01959430567333863,x_axis,y_axis))
print(func([2547,3,3,60],[ 0.26101297, -0.12314478, -0.14273909, -0.50344429],0.01959430567333863))