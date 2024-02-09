import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

d = fetch_california_housing()
X, y = d.data, d.target

# Разделение данных на тренировочную и тестовую выборки. Используйте соотношение 80/20
#
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=1)

class MyDataset(Dataset):
  #
  # Определите конструктор и методы __getitem__ и __len__
  # Сделайте всё в точности так, как мы делали в видео
  #
    def __init__(self, X,y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.X.shape[0]

# Инициализируем тренировочный torch.dataset
train_dataset = MyDataset(X_train, y_train)

# Преобразуем тестовую выборку в torch-тензоры всю целиком
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

# Пусть размер батча у нас равен 128
batch_size = 128

# Создайте Dataloader для тренировочной выборки на основе экземпляра train_dataset
# Делайте как в видео, размер батча – batch_size
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

class RegressionNet(nn.Module):
  #
  # ВАШ КОД ЗДЕСЬ
  #
    def __init__(self):
            super().__init__()
            self.hidden1 = nn.Linear(8, 24, bias=True)
            self.f1 = nn.ReLU()
            self.hidden2 = nn.Linear(24,12,bias=True)
            self.f2 = nn.ReLU()
            self.hidden3 = nn.Linear(12,6, bias=True)
            self.f3 = nn.ReLU()
            self.output = nn.Linear(6,1,bias=True)

    def forward(self, x):
            x = self.f1(self.hidden1(x))
            x = self.f2(self.hidden2(x))
            x = self.f3(self.hidden3(x))

            return self.output(x)


# Объявляем экземпляр класса нейронной сети
model = RegressionNet()

loss_fn = nn.MSELoss()

# Создаём оптимизатор. Тут будем использовать вариацию стохастического
# градиентного спуска Adam. Это адаптивный алгоритм, который выбирает
# шаг изменения весов (learning rate) в зависимости от текущей ситуации.
# Это очень эффективный алгоритм, который в большинстве случаев работает
# лучше, чем обычный градиентный спуск с постоянным шагом. В этой задаче –
# точно лучше.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Делаем 100 эпох
num_epochs = 100

# Сюда будем сохранять значение функции потерь на тестовой выборке
# после каждой эпохи обучения
loss_test = []

# Реализуйте тренировочный цикл
for i in range(num_epochs):
    for X, y in train_dataloader:
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # Конец эпохи: считаем функцию потерь на тестовой выборке,
        # сохраняем в список, чтобы потом нарисовать график
        loss = loss_fn(
            model(X_test_tensor),
            y_test_tensor.unsqueeze(-1)
        ).item()
        loss_test.append(loss)
        print(f'epoch {i} loss {loss}')


step = np.arange(0, num_epochs)

fig, ax = plt.subplots(figsize=(8,5))

# Рисуем зависимость ошибки от эпохи обучения
plt.plot(step, np.array(loss_test))

plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()