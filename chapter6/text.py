# 名字：李志翔
# 创建时间:2021/7/31 17:51
#2D函数优化实战
import matplotlib.pyplot as plt
import numpy as np
import torch

def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2


x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
X,Y=np.meshgrid(x,y)
Z=himmelblau([X,Y])

fig=plt.figure('himmelblau')
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()                                     #这里是把函数的图像画出来


x=torch.tensor([0.,0.],requires_grad=True)     #这里是对函数进行优化
optimizer=torch.optim.Adam([x],lr=1e-3)
for step in range(20000):
    pred=himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    if step%2000 ==0:
        print('step {}:x={},f(x)={}'.format(step,x.tolist(),pred.item()))