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
    sigma=np.std(x,axis=0)
    x_normal=(x-mu)/sigma
    return x_normal,sigma,mu
def func(x,w,b):
    n=np.dot(w,x)+b
    return n
#cost function for the program
def func_cost(w,b,x,y):
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
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(nums):
        jw,jb=gradient(x,b,w,y)
        b=b-alpha*jb
        w=w-alpha*jw
    return w,b
x_normal,sigma,mu = z_normalization(x_axis)
w_final, b_final = gradient_desent(w,0,x_normal,y_axis,100000,1.1e-3)
print("Final weights and bias:",'w:', w_final,'b:', b_final)
print("Final cost:", func_cost(w_final,b_final,x_normal,y_axis))
x_new = np.array([2547,3,3,60])
x_new_norm = (x_new - mu)/sigma
print("Prediction:", func(x_new_norm,w_final,b_final))