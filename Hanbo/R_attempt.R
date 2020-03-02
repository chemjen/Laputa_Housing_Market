data <- read.csv('train_cleaned.csv')
data1 <- read.csv('test_cleaned.csv')
fit <- lm(log1p(SalePrice) ~ ., data = data)
summary(fit)
predict.lm(fit, newdata = data1)
empty.model = lm(SalePrice ~ 1, data = data)
scope = list(lower = formula(empty.model) , upper = formula(fit))
library(MASS)
forwardAIC = step(empty.model, scope, direction = "both", k = 2)

data1 <- read.csv('test_cleaned.csv')
head(data1)

summary(forwardAIC)

log1p(data$SalePrice)

set.seed(27042018)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=data, y=all$SalePrice[!is.na(all$SalePrice)], method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune