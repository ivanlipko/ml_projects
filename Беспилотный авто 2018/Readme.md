# Открытый Кубок России по программированию беспилотного автомобиля. Осень-зима 2018 - январь 2019

Организаторы https://vk.com/avt.global, Академия Высоких Технологий - Лидеры инженерного образования России. 

**Решение двух задач:**
- распознавание сигналов светофора
- классификация дорожных знаков

Обучаем классификаторы на данных. Проверяем разные классификаторы (SVM, RandomTree, ...) и гипотезы признаков (цвет, HOG, SIFT).

Файлы *_dev.py -- этот модули для исследования. Аналогично теже самые модули без суффикса будут работать на сервере.

*Поскольку на сервер проверки отправляем код, который там будет выполняться, то: обучение я провёл у себя на машине. Сохранил калссификатор в сам файл скрипта, а далее открыл-загрузил методами in-memory.*

Результат: получил хорошее решение на SVM+HOG, что дало на лидерборде оценку 1.00 точности.

Раньше здесь была таблица участников: http://sim.newgen.education/mcv/3

