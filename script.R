# зчитуємо дані
data <- read.table('C:\\Users\\Razor\\Desktop\\дистанційне навчання\\статистичні алгоритми навчання\\lab2\\data3.csv',
                   header = T, sep = ',')

col <- c('red', 'blue') # задамо палітру кольорів
# виведемо діграму розсіювання х1 і х2 із розфарбуванням
plot(data[,1],data[,2], col = col[as.factor(data$y)], xlab = 'x1', ylab = 'x2') 
legend('topleft', legend = c('0 class', '1 class'), col = col, pch = 1) # виведемо легенду на діаграму

# задамо параметри кількості нейронів на кожному шарі, а також к-сть спостережень
N <- length(data[,1])
neurons <- c(2,4,1)

# задамо функцію, що генеруватиме W^1, W^2, b^1 та b^2
generator <- function(n){
  w1 <- rnorm(n[1]*n[2], mean = 0, sd = 0.01)
  W1 <- matrix(data = w1, nrow = n[1], ncol = n[2])
  w2 <- rnorm(n[2]*n[3], mean = 0, sd = 0.01)
  W2 <- matrix(data = w2, nrow = n[2], ncol = n[3])
  b1 <- matrix(data = rep(0, n[2]), nrow = n[2])
  b2 <- matrix(data = rep(0, n[3]), nrow = n[3])
  res <- list('W1' = W1, 'W2' = W2, 'b1' = b1, 'b2' = b2)
  return(res)
}

param <- generator(neurons)

# реалізуємо прогноз за допомогою forward propagation
library(e1071)
frw <- function(d, n, N, param){
  c <- param
  B1 <- matrix(data = rep(c$b1, N), nrow = N)
  Z1 <- as.matrix(d[,1:2]) %*% c$W1 + B1
  A1 <- tanh(Z1)
  B2 <- matrix(data = rep(c$b2, N), nrow = N)
  Z2 <- as.matrix(A1) %*% c$W2 + B2
  A2 <- sigmoid(Z2)
  res <- list('A2' = A2, "Z2" = Z2, "A1" = A1, 'Z1' = Z1)
  return(res)
}

frw_res <- frw(data, neurons, N, param)

# визначимо цільову функцію
cross_entopy_loss <- function(d, n, N, param){
  c <- frw(d, n, N, param)
  c <- c$A2
  res <- 0
  for(j in 1:N){
    r <- d[j,3]*log(c[j]) + (1-d[j,3])*log(1-c[j])
    res <- res + r
  }
  return(-res/N)
}

# реалізуємо алгоритм backward-propagation
bcw <- function(d, n, N, frw_res, gen_res){
  c <- frw_res
  g <- gen_res
  d2 <- c$A2 - as.matrix(d[,3])
  w2 <- (1/N)*t(d2) %*% c$A1
  b2 <- sum(d2)/N
  d1 <- d2 %*% t(g$W2) * (1-(c$A1)^2)
  w1 <- (1/N) * t(d1) %*% as.matrix(d[,1:2])
  b1 <- matrix(sum(d1)/N, nrow = n[2])
  res <- list('dW2' = w2, 'dW1' = w1, 'db2' = b2, 'db1' = b1)
  return(res)
}

bcw_res <- bcw(data, neurons, N, frw_res, param)

# покращимо параметри моделі
impr_par <- function(alpha, param, bcw_res){
  a <- param
  b <- bcw_res
  w1 <- t(a$W1) - alpha * b$dW1
  w2 <- t(a$W2) - alpha * b$dW2
  b1 <- a$b1 - alpha * b$db1
  b2 <- a$b2 - alpha * b$db2
  res <- list('W2' = w2, "W1" = w1, 'b2' = b2, 'b1' = b1)
  return(res)
}

impr <- impr_par(0.05, param, bcw_res)

# реалізуємо функцію що тренуватиме модель і підбиратиме оптимальний параметр
final <- function(d, n, n_iter, alpha){
  p <- generator(n)
  loss <- c()
  for(j in 1:n_iter){
    frw_res <- frw(d, n, N = 600, param = p)
    loss <- c(loss, cross_entopy_loss(d, n, N = length(d[,1]), p))
    bcw_res <- bcw(d, n, N = length(d[,1]), frw_res, p)
    p1 <- impr_par(alpha, p, bcw_res)
    p1$W1 <- t(p1$W1)
    p1$W2 <- t(p1$W2)
    p <- p1
    if(j %% 500 == 0)cat('Iteration',j,'|CE_loss:',loss[j],'\n')
  }
  res <- list('new_parameters' = p1, 'loss_values' = loss)
  return(res)
}

# проженемо алгоритм 30 000 разів для параметра alpha = 0.01
res2 <- final(data, neurons, 30000, 0.1)

# побудуємо оптимальну модель
res_frw2 <- frw(data, neurons, N, param = res2$new_parameters)
final_pred2 <- res_frw2$A2
final_pred2 <- round(final_pred2)

# виведемо таблицю спряженості
table(final_pred2, data[,3])
print(sum(final_pred2 == data[,3])/N)

# виведемо діаграму розсіювання з передбаченим розмалюванням
plot(data[,1], data[,2], col = col[final_pred2+1])
print(sum(final_pred2 == data[,3])/N) # точність роботи класифікатора
