import numpy as np

# Hypothsis : A*X = b
"""
A = [   [ 0.9978,-0.0656  , -0.0111,-433.1939912651718],
   [0.0311    ,0.6071  ,-0.7940,243.2401730555002],
    [0.0589  , 0.7919     , 0.6078,2627.259807194887],
        [0,0,0,1]];


B = [[3832.28668083377	,0,	1985.50957545762,0],
      [0,	3704.06873556086	,912.542650211984,0],
      [0,	0	,1,0]]
C = np.dot(B,A)
D =[[0],[0],[0],[1]]
E = np.dot(C,D)
print(C)
print(E)
i = 3
while i>0:
    i = i -1
    j=4
    while j>0:
        j=j-1
        C[i][j] = C[i][j]/2.62725981e+03
print(C)
print('-------------------------------------------')
D=[[1.49996668e+00 , 5.02777465e-01],
   [6.43047555e-02 , 1.13098166e+00]]
b=[[2597-1.35362553e+03],
   [557-1.25547726e+03]]


#[or] b = np.transpose(np.array([-3,5,-2]))# 转置


# 求解未知参数矩阵X
X = np.linalg.solve(D,b) # 方式一：直接使用numpy的solve函数一键求解
#A_inv=np.linalg.inv(A) # 方式二：先求逆运算，再点积求值
#X=np.dot(A_inv,b) # a.dot(b) 与 np.dot(a,b) 效果相同；but np.dot(a,b)与np.dot(b,a)效果肯定是不同的(线性代数/矩阵常识)
print("方程组的解：\n",X);
print('-------------------------------------------------------------------------------')


A = [[0.9990 ,   0.0007   , 0.0455   , -4.302938544781178e+02],
   [ 0.0439  ,  0.2498  , -0.9673  , 1.193733343859837e+03],
  [ -0.0121  ,  0.9683  ,  0.2496 , 3.981190880081052e+03],
     [0, 0, 0, 1]]

B = [[1795.59711915637	,0,	985.021908172139,0],
[0	,1819.06343307926	,454.759468038578,0],
[0	,0	,1,0]]

C = np.dot(B,A)
D =[[0],[0],[0],[1]]
E = np.dot(C,D)
print(C)
print(E)
i = 3
while i>0:
    i = i -1
    j=4
    while j>0:
        j=j-1
        C[i][j] = C[i][j]/3.98119088e+03
print(C)
print('-------------------------------------------')
D=[[4.47575314e-01, 2.39891445e-01],
   [1.86763954e-02 , 2.24743215e-01]]
b=[[800-7.90950730e+02],
   [1004-1.00019342e+03]]


# 求解未知参数矩阵X
X = np.linalg.solve(D,b) # 方式一：直接使用numpy的solve函数一键求解
#A_inv=np.linalg.inv(A) # 方式二：先求逆运算，再点积求值
#X=np.dot(A_inv,b) # a.dot(b) 与 np.dot(a,b) 效果相同；but np.dot(a,b)与np.dot(b,a)效果肯定是不同的(线性代数/矩阵常识)
print(X[0][0]/1000)
print(X[1][0]/1000)
print("方程组的解：\n",X);
print('-------------------------------------------------------------------------------')
"""
#最大的那个 一号机位
def celiang(xzuobiao,yzuobiao):
    A = [[0.9999, 0.0011, 0.0172, -89.6842190842247],
         [0.0158, 0.3460, -0.9381, 841.685588141010],
         [-0.0070, 0.9382, 0.3459, 4068.18666704540],
         [0, 0, 0, 1]]
#外参数
    A = [[0.99951426, 0.01815799, -0.02532834, -691.32894494],
         [0.01750038, 0.34546335, 0.93826905, 592.5752263],
         [0.0257871, -0.93825655, 0.34497777, 4596.52120875],
         [0, 0, 0, 1]]
#内参数
    B = [[1789.37421797427, 0, 976.452625964358, 0],
         [0, 1787.78799529300, 548.228414477184, 0],
         [0, 0, 1, 0]]
    C = np.dot(B, A)
    #print(C)

   # D = [[-7.00000000e-03 - 1.78236011e+03 / xzuobiao, -9.38200000e-01 - 9.18076165e+02 / xzuobiao],
    #     [-7.00000000e-03 - 2.44094514e+01 / yzuobiao, -9.38200000e-01 - 1.13292254e+03 / yzuobiao]]
   # b = [[3.81191292e+06 / xzuobiao - 4.06818667e+03],
   #      [3.73505092e+06 / yzuobiao - 4.06818667e+03]]
    D = [[C[2][0] - C[0][0] / xzuobiao, C[2][1] - C[0][1] / xzuobiao],
         [C[2][0] - C[1][0] / yzuobiao, C[2][1] - C[1][1] / yzuobiao]]
    b = [[C[0][3] / xzuobiao - C[2][3]],
         [C[1][3] / yzuobiao - C[2][3]]]

    # 求解未知参数矩阵X
    X = np.linalg.solve(D, b)  # 方式一：直接使用numpy的solve函数一键求解
    # A_inv=np.linalg.inv(A) # 方式二：先求逆运算，再点积求值
    # X=np.dot(A_inv,b) # a.dot(b) 与 np.dot(a,b) 效果相同；but np.dot(a,b)与np.dot(b,a)效果肯定是不同的(线性代数/矩阵常识)
    resx = X[0][0] / 1000
    resy = X[1][0] / 1000
    # print(X[0][0] / 1000)
    # print(X[1][0] / 1000)
    # print("方程组的解：\n", X);
    # print('-------------------------------------------------------------------------------')
    return resx,-resy

#2号机位 实验室门口的
def celiang2(xzuobiao,yzuobiao):
#外参数
    A = [[0.99993201 , 0.00845173 , 0.00803355, -998.0521219],
         [-0.01002735 , 0.27160349 , 0.962357, 1027.20578846],
         [0.00595164, -0.96237213 , 0.27166978, 4289.38246959],
         [0, 0, 0, 1]]
#内参数
    B = [[1789.37421797427, 0, 976.452625964358, 0],
         [0, 1787.78799529300, 548.228414477184, 0],
         [0, 0, 1, 0]]
    C = np.dot(B, A)
    #print(C)

   # D = [[-7.00000000e-03 - 1.78236011e+03 / xzuobiao, -9.38200000e-01 - 9.18076165e+02 / xzuobiao],
    #     [-7.00000000e-03 - 2.44094514e+01 / yzuobiao, -9.38200000e-01 - 1.13292254e+03 / yzuobiao]]
   # b = [[3.81191292e+06 / xzuobiao - 4.06818667e+03],
   #      [3.73505092e+06 / yzuobiao - 4.06818667e+03]]
    D = [[C[2][0] - C[0][0] / xzuobiao, C[2][1] - C[0][1] / xzuobiao],
         [C[2][0] - C[1][0] / yzuobiao, C[2][1] - C[1][1] / yzuobiao]]
    b = [[C[0][3] / xzuobiao - C[2][3]],
         [C[1][3] / yzuobiao - C[2][3]]]

    # 求解未知参数矩阵X
    X = np.linalg.solve(D, b)  # 方式一：直接使用numpy的solve函数一键求解
    # A_inv=np.linalg.inv(A) # 方式二：先求逆运算，再点积求值
    # X=np.dot(A_inv,b) # a.dot(b) 与 np.dot(a,b) 效果相同；but np.dot(a,b)与np.dot(b,a)效果肯定是不同的(线性代数/矩阵常识)
    resx = X[0][0] / 1000
    resy = X[1][0] / 1000
    # print(X[0][0] / 1000)
    # print(X[1][0] / 1000)
    # print("方程组的解：\n", X);
    # print('-------------------------------------------------------------------------------')
    return resx,-resy