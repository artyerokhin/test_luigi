## Test Luigi python project
### Тестовый учебный проект по работе с Luigi

**Luigi** - это один из немногих инструментов в экосистеме Python для построения т.н. pipeline’ов или, по-простому, выполнения пакетных задач (batch jobs). Разработан был инженерами из Spotify.

### В данном проекте рассматривается простой pipeline работы с данными на примере:
- загрузки данных uber api по оценке случайных пар место старта - место конца поездки
- сохранения указанных данных
- предсказания на основе указанных данных времени поездки на основании данных uber
- проверка поведения luigi (для этого добавлен файл luigi_script_break.py)

### Итак, для того, чтобы запустить:
1. устанавливаем luigi (pip install luigi)
2. запускаем демона luigi (luigid), на localhost:8082 запускается веб-морда для просмотра процесса работы
3. получаем secret для Uber API (описание ниже)
4. после сохраняем секрет в config.py и запускаем PYTHONPATH=./ luigi --module luigi_script_break TrainModel --GetApiData-n 15
5. После шага 4 мы обязательно получим ошибку, что естесственно. Нужно это для того, чтобы показать, что при падении у нас не перезапускаются выполненные части pipeline. Падение и ошибку можно посомтреть на localhost:8082
6. Далее запускаем уже обычный скрипт PYTHONPATH=./ luigi --module luigi_script TrainModel --GetApiData-n 15
7. Наслаждаемся полученным примером

### Как получить ключ uber API:
1. регистрируем свое приложение uber на https://developer.uber.com/dashboard/ (личный кабинет - новое приложение)
2. uri и url перенаправления можно задать как http://127.0.0.1
3. проставляем доступ ко всему, кроме того, что требует "полный доступ"
4. в графе ТЕСТ С ЛИЧНЫМ КЛЮЧОМ ДОСТУПА кликаем по "создать ключ доступа"
5. полученный ключ копируем себе
6. примерный объем возможных запросов ~ 100 в час, [ссылка на доки](https://developer.uber.com/docs/riders/introduction)

### Примерный план применения такого пайплайна:
1. Получаем данные для случайных (или неслучайных) поездок за каждый час. К примеру, поставив скрипт в cron
2. Складируем полученные данные в базу данных
3. Копим результаты, постепенно улучшая модель (для этого добавилено дообучение). Впрочем, модель взять просто для примера
4. Копим результаты оценки качества модели на кросс-валидации
5. В итоге, получаем ouput в виде пополняющейся БД и обновленной модели для каждого нового набора данных

### Описание файловой структуры:
- data - файлы данных для каждого скачивания из API
- log - файлы виртуальных логов (используются, чтобы следить за output в случае записи в БД)
- model - файлы моделей (pkl-файлы)
- validation - файлы валидации (предсказаний на k-fold разбиении)
- config.py - файл конфига
- nodes.csv - файл нод для генерации случайных пар точка старта - точка конца
- luigi_script_break.py - основной скрипт pipeline (со специально внесенной ошибкой)
- luigi_script.py - основной скрипт pipeline (без специально внесенной ошибки)
- test_uber_api.py - некоторый код, использующийся в pipeline
- uber.sqlite (генерируется впоследствии) - sqlite БД для таблиц полученных из API данных и результатов предсказаний

### Ссылки по теме:
1. [Luigi official documentation](http://luigi.readthedocs.io/en/stable/index.html)
2. [Complex Luigi pipelines](https://www.promptworks.com/blog/configuring-complex-luigi-pipelines)
3. [Blogpost about Luigi](https://marcobonzanini.com/2015/10/24/building-data-pipelines-with-python-and-luigi/)
4. [Another blogpost](https://khashtamov.com/ru/data-pipeline-luigi-python/)
5. [Luigi git repo](https://github.com/spotify/luigi)
