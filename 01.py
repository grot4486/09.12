def mult_const(A, k): # умножение константы на матрицу
    M = len(A) # кол-во строк
    N = len (A[0]) # кол-во столбцов
    result = [[1]*N for i in range(M)]
    for i in range(M):
       for j in range(N):
           result[i][j] = A [i][j] * k
           return result
           
def proizv(A,B): # поэлементное умножение двух матриц
    M = len(A)
    N = len (A[0])
    C = [[0]*N for i in range(M)]
    for i in range(M):
        for j in range(N):
            C[i][j] = A[i][j] * B[i][j]
    return C
   
def sub_from_const(A,k): # поэлементное вычитание матрицы из константы
    M = len(A)
    N = len (A[0])
    C = [[0]*N for i in range(M)]
    for i in range(M):
        for j in range(N):
            C[i][j] = k - A[i][j]
    return C
   
def mult_matrix(A, B): # векторное произведение2-х матриц
    A_Rows = len(A) # кол-во строк A
    A_Cols = len(A[0]) # кол-во столбцов А
    B_Rows = len(B) # кол-во строк В
    B_Cols = len(B[0]) # кол-во столбцов В
   
    if (A_Cols != B_Rows): # проверка размеров матриц
      print ('кол-во столбцов 1-й матрицы не равно кол-ву сторк 2-ой')
      return
 
    С = [[0 for rw in range (B_Cols)] for col in range(A_Rows)]
    for i in range(A_Rows):
      for j in range(B_Cols):
          for k in range(A_Cols):
              C[i][j] += A[i][k] * B[k][j]
    return C
   
def transpose (A): # транспонирование матрицы
    M = len(A) # кол-во сторк
    N = len(A[0]) # кол-во столбцов
    A_t = [[0]*M for i in range(N)]
    for i in range(N):
        for j in range(M):
            A_t[i][j] = A [j][i]
    return A_t
   
def summ_matrix(A,B):# поэлементное сложение двух матриц
    M = len(A)
    N = len(A[0])
    C = [[0]*N for i in range(M)]
    for i in range(M):
        for j in range(N):
            C[i][j] = A[i][j] + B[i][j]
    return C
   
   
def sigma(x): # сигмоида
    return 1 / (1 + math.exp(-x))
   
def f_activate(X):
    for i in range(len(X)):
        X[i][0] = sigma(X[i][0])
    return X
   
def train(self, inputs_list, targets_list):
    O_h = f_activate(mult_matrix(self.W_i_h, inputs_list)) # выход скрытого слоя
    O = f_activate(mult_matrix(self.W_h_o, O_h)) # выход выходного слоя
   
    E_o = sub_matrix(targets_list, O) # ошибка выходного слоя
    E_h = mult_matrix(transpose(self.W_h_o), E_o) # ошибкак скрытого слоя
   
    dW_h_o = [[0]*self.input_nodes for i in range(self.hidden_nodes)] # изменение матрицы выходного слоя
    dW_h_o = mult_const(mult_matrix(proizv(proizv(E_o, O), sub_from_const(O, 1)), transpose(O_h)), self.learning_rate)
    self.W_h_o = summ_matrix(self.W_h_o , dW_h_o)
    dW_i_h = [[0]*self.input_nodes for i in range(self.hidden_nodes)] # изменение матрицы сктрытого слоя
    dW_i_h = mult_const(mult_matrix(proizv(proizv(E_h, O_h), sub_from_const(O_h, 1)), transpose(inputs_list)), self.learning_rate)
    self.W_i_h = summ_matrix(self.W_i_h, dW_i_h)
