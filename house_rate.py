import numpy as np
#flow chart (input and output)
y_axis=np.array([100,200,300,400,500])
# y_axis=np.array([300,500])
x_axis=np.array([1,2,3,4,5])
# x_axis=np.array([1,2])
def func(x,w,b):
    n=w*x+b
    return n
#cost function for the program
def func_cost(w,b,x,y):
    l=x.shape[0]
    m=np.zeros(l)
    cost_sum=0
    for i in range(l):
        n=w*x[i]+b
        m=(n-y[i])**2
        cost_sum=cost_sum+m
    total_sum=(1/(2*l))*cost_sum
    return total_sum
#gradient desent
def gradient(x,b,w,y):
    temp_w=w
    temp_b=b
    db=0
    dw=0
    l=x.shape[0]
    for i in range(l):
        fun=w*x[i]+b
        kw=(fun-y[i])*x[i]
        kb=(fun-y[i])
        db=kb
        dw=kw
    return dw,db
# final gradient desent function
def gradient_desent(w,b,x,y,nums,alpha):
    w=w
    b=b
    for i in range(nums):
        i,j=gradient(x,b,w,y)
        b=b-alpha*j
        w=w-alpha*i
    return w,b
print(gradient_desent(0,0,x_axis,y_axis,10000,1.0e-2))
print(func_cost(96.15384615384613,19.23076923076923,x_axis,y_axis))
print(func(4.5,96.15384615384613,19.23076923076923))