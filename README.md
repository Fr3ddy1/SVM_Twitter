# SVM_Twitter

Uso del modelo SVM para analizar data en Twitter

Support Vector Machine en R
---------------------------

EN este ejemplo se mostrará el uso del modelo SVM en R, el cuál se
aplicará a una data obtenida de Twitter con el fin de hacer una
clasificación. La misma snos indicará y nos ayudará a determinar si la
opinión de un tweet en específico es positiva ó negativa.

    #cargo librerias
    #cargo librerias
    library("devtools")
    library("tm")

    ## Loading required package: NLP

    #paquetes de machine learning
    library(caTools)
    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:NLP':
    ## 
    ##     annotate

    library(e1071)

    ## Warning: package 'e1071' was built under R version 3.5.2

    #nube de palabras
    library(wordcloud)

    ## Loading required package: RColorBrewer

La base de datos a usar será "tweets.csv", la cual se obtuvo luego de
una consulta a través de Google Drive, la misma muestra la información
obtenida al realizar una busqueda sobre "Apple". Cabe destacar que esta
consulta también se pudo haber hecho mediante la API de Twitter, en este
caso el límite de los tweets a obtener ronda los 18.000 cada 15 minutos.

    #leo base de datos de Tweets que baje usando google Drive
    tweets <- read.csv(paste0(getwd(),"/Data/tweets.csv"))

    #tomo 1era fila para colocar nombre a columnas
    a <- tweets[1,]
    for(i in 1:length(a)){
      a[[i]] <- as.character(a[[i]])
    }

    #pongo nombre a columnas
    names(tweets) <- a
    tweets <- tweets[-1,]

    data <- data.frame(texto=as.character(tweets$`Tweet Text`))

    head(data)

    ##                                                                                                                                                                                                                                  texto
    ## 1                                                                                                                                                                                          Die neuen Apple Zahlen sind da – alle Infos
    ## 2                                                                                                                                                                                      Top five MJ song \U0001f60c\U0001f60c\U0001f60c
    ## 3                                                                                                                                                                                                                                     
    ## 4                                                                                                                                                                                             C’est le meilleur son de son album frère
    ## 5 ** THE #APPLE RULES ** $AAPL Q3 19 Earnings: - Revenue: $53.8B (exp $53.35B) - EPS: $2.18 (exp $2.10) - iPhone Revenue: $25.99B (exp $26.54B) - Services Revenue: $11.46B (exp $11.88B) - Sees Q4 Revenue $61B To $64B (exp $61.04B)
    ## 6

Una vez obtenida la data es necesario pre-procesarla con el fin de
obtener una clasificación positiva (1) o negativa (-1) a cada Tweet, a
modo de ejemplo y con el fin de mostrar como funciona el modelo se
asignará al azar un valor a cada tweet obtenido.

    #creo columna de 1 y -1 
    f <- rbinom(length(data$texto),1,0.7)
    f[which(f==0)] <- -1

    data$sent <- f 

    table(data$sent)

    ## 
    ##   -1    1 
    ##  981 2321

Para empezar se debe transformar el texto de cada tweet a un formato
Corpus, y de la manera siguiente podemos acceder a su información,

    #preprocesamiento
    #convierto texo del tweet a objeto corpus
    corpus <- Corpus(VectorSource(data$texto))
    length(corpus)

    ## [1] 3302

    #ingreso a la informacion en el objeto
    content(corpus[[20]])

    ## [1] "Always enjoy #olderandwider but particularly enjoyed \u2066@jennyeclair\u2069 and \u2066@greyprideuk\u2069 chatting to \u2066@ArabellaWeir\u2069 tonight so sad so funny. Wonder if show will be in London after Edinburgh? If so I’m there."

El primer cambio a realizar, es cambiar todas la letras mayusculas a
minusculas, esto se logra mediante los siguientes comandos,

    #cambio letras mayusculas a minusculas
    corpus1 <- tm_map(corpus,tolower)

    ## Warning in tm_map.SimpleCorpus(corpus, tolower): transformation drops
    ## documents

    #realizo transformacion de data anterios
    corpus1 <- tm_map(corpus1,PlainTextDocument)

    ## Warning in tm_map.SimpleCorpus(corpus1, PlainTextDocument): transformation
    ## drops documents

    #accedo a informacion del nuevo objeto
    content(corpus1[[1]])[20]

    ## [1] "always enjoy #olderandwider but particularly enjoyed \u2066@jennyeclair\u2069 and \u2066@greyprideuk\u2069 chatting to \u2066@arabellaweir\u2069 tonight so sad so funny. wonder if show will be in london after edinburgh? if so i’m there."

Luego de esto se procede a quitar signos de puntuación,

    #quito puntuacion
    corpus1 <- tm_map(corpus1,removePunctuation)

    #accedo a informacion del nuevo objeto
    content(corpus1[[1]])[20]

    ## [1] "always enjoy olderandwider but particularly enjoyed \u2066jennyeclair\u2069 and \u2066greyprideuk\u2069 chatting to \u2066arabellaweir\u2069 tonight so sad so funny wonder if show will be in london after edinburgh if so i’m there"

Otro aspecto importante a considerar son las "stop words", las cuales
son palabras que se usan como conectores y no representan o no nos dan
información relevante, por tal motivo las mismas se eliminarán del texto
de cada tweet. Existen "stop words" tanto en ingles como en español y
las mismas ya se encuentran prefijadas en los paquetes instalados,

    #quito stop words
    #primeras palabras
    stopwords("english")[1:10]

    ##  [1] "i"         "me"        "my"        "myself"    "we"       
    ##  [6] "our"       "ours"      "ourselves" "you"       "your"

    stopwords("spanish")[1:10]

    ##  [1] "de"  "la"  "que" "el"  "en"  "y"   "a"   "los" "del" "se"

    #quito stopwords
    #existen problemas con caracter "’", lo quite manualmente
    corpus1 <- tm_map(corpus1,removeWords,c(stopwords("english"),"apple","’"))

    #accedo a informacion del nuevo objeto
    content(corpus1[[1]])[20]

    ## [1] "always enjoy olderandwider  particularly enjoyed \u2066jennyeclair\u2069  \u2066greyprideuk\u2069 chatting  \u2066arabellaweir\u2069 tonight  sad  funny wonder  show will   london  edinburgh   m "

Finalmente, se debe cambiar cada palabra a su forma base o raíz esto es
cada palabra o verbo que esté conjugado se debe pasar a su forma base o
verbo.

    #modifico palabras a su raiz
    corpus1 <- tm_map(corpus1,stemDocument)

    #accedo a informacion del nuevo objeto
    content(corpus1[[1]])[20]

    ## [1] "alway enjoy olderandwid particular enjoy \u2066jennyeclair\u2069 \u2066greyprideuk\u2069 chat \u2066arabellaweir\u2069 tonight sad funni wonder show will london edinburgh m"

Una vez realizado este proceso de limpieza se procede a crear una matriz
de frecuencias, la cual nos va a proporcionar información sobre la
frecuencia de uso de cada palabra,

    #creo matriz de frecuencias
    #creo nuevo corpus con data limpia
    corpus2 <- Corpus(VectorSource(as.character(content(corpus1[[1]]))))

    frecuencies <- DocumentTermMatrix(corpus2)

    #veo matriz
    inspect(frecuencies[800:805,505:515])

    ## <<DocumentTermMatrix (documents: 6, terms: 11)>>
    ## Non-/sparse entries: 0/66
    ## Sparsity           : 100%
    ## Maximal term length: 9
    ## Weighting          : term frequency (tf)
    ## Sample             :
    ##      Terms
    ## Docs  1240pm foxbusi get pacif readi tune despit thing actualiza algún amo
    ##   800      0       0   0     0     0    0      0     0         0     0   0
    ##   801      0       0   0     0     0    0      0     0         0     0   0
    ##   802      0       0   0     0     0    0      0     0         0     0   0
    ##   803      0       0   0     0     0    0      0     0         0     0   0
    ##   804      0       0   0     0     0    0      0     0         0     0   0
    ##   805      0       0   0     0     0    0      0     0         0     0   0

Debido a la forma de esta matriz la misma contará con una gran cantidad
de ceros, por tal motivo usaremos el comando "removeSparseTerms" para
remover y no considerar esta gran cantidad de datos, esto se logra
mediante los siguientes comandos,

    #encuento frecuencia
    findFreqTerms(frecuencies,lowfreq = 50)

    ##  [1] "song"          "210"           "218"           "aapl"         
    ##  [5] "earn"          "iphon"         "revenu"        "see"          
    ##  [9] "servic"        "via"           "will"          "year"         
    ## [13] "2019"          "like"          "third"         "work"         
    ## [17] "digit"         "make"          "itun"          "love"         
    ## [21] "podcast"       "want"          "news"          "ipad"         
    ## [25] "per"           "one"           "watch"         "quarter"      
    ## [29] "trade"         "billion"       "stock"         "now"          
    ## [33] "can"           "spotifi"       "trump"         "report"       
    ## [37] "result"        "sale"          "angel"         "california"   
    ## [41] "elect"         "law"           "los"           "new"          
    ## [45] "requir"        "return"        "tax"           "time"         
    ## [49] "just"          "que"           "—"             "microsoft"    
    ## [53] "app"           "appl"          "music"         "beat"         
    ## [57] "expect"        "get"           "free"          "googl"        
    ## [61] "est"           "share"         "today"         "listen"       
    ## [65] "download"      "access"        "data"          "health"       
    ## [69] "patient"       "standard"      "test"          "2019年7月31日"
    ## [73] "sleepmeist"    "時刻"          "play"          "store"        
    ## [77] "android"

    #reduzco matriz
    sparse <- removeSparseTerms(frecuencies,0.995)
    sparse

    ## <<DocumentTermMatrix (documents: 3302, terms: 246)>>
    ## Non-/sparse entries: 10577/801715
    ## Sparsity           : 99%
    ## Maximal term length: 10
    ## Weighting          : term frequency (tf)

    #convierto matriz a dataframe
    tweetsSparse <- as.data.frame(as.matrix(sparse))

    colnames(tweetsSparse) <- make.names(colnames(tweetsSparse))
    tweetsSparse$sentiment <- data$sent

Una vez eliminado esta data, se procederá a dividir la misma en dos, una
data de entrenamiento y una data de prueba, esto se realiza mediante el
comando "sample.split", donde le decimos que queremos generar unos
indices de manera tal que el 80% sea data de entrenamiento y el 20%
restante sea data de prueba,

    #maquina de soporte vectorial
    #dividir data en un 80% entrenamiento y un 20% para evaluar

    set.seed(12)

    split <- sample.split(tweetsSparse$sentiment,SplitRatio = 0.8)

    #creo data de entrenamiento y data de evaluacion
    trainSparse <- subset(tweetsSparse,split==TRUE)
    testSparse <- subset(tweetsSparse,split==FALSE)
    testSparse$sentiment <- as.factor(testSparse$sentiment)

    table(testSparse$sentiment)

    ## 
    ##  -1   1 
    ## 196 464

De esta manera uso la data de entrenamiento "trainSparse", para correr y
calibrar el modelo, esto usando la funcion "svm", posteriormente se
muestra un resumen del modelo,

    #ejecuto modelo
    SVM <- svm(as.factor(sentiment)~.,data = trainSparse)

    summary(SVM)

    ## 
    ## Call:
    ## svm(formula = as.factor(sentiment) ~ ., data = trainSparse)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ##       gamma:  0.004065041 
    ## 
    ## Number of Support Vectors:  2057
    ## 
    ##  ( 1272 785 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  -1 1

Luego de obtener el modelo, se pueden realizar predicciones usando la
data de prueba "testSparse", para luego evaluar los resultados y generar
una matriz de confusión,

    #realizo predicciones
    predictSVM <- predict(SVM,newdata = testSparse)

    #realizo prueba de la prediccion
    confusionMatrix(predictSVM,testSparse$sentiment)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  -1   1
    ##         -1   4   6
    ##         1  192 458
    ##                                           
    ##                Accuracy : 0.7             
    ##                  95% CI : (0.6634, 0.7348)
    ##     No Information Rate : 0.703           
    ##     P-Value [Acc > NIR] : 0.5865          
    ##                                           
    ##                   Kappa : 0.0103          
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.020408        
    ##             Specificity : 0.987069        
    ##          Pos Pred Value : 0.400000        
    ##          Neg Pred Value : 0.704615        
    ##              Prevalence : 0.296970        
    ##          Detection Rate : 0.006061        
    ##    Detection Prevalence : 0.015152        
    ##       Balanced Accuracy : 0.503739        
    ##                                           
    ##        'Positive' Class : -1              
    ## 

En este caso podemos apreciar que el modelo tiene una precisión de
"0.6838", lo que indica que no es muy bueno, sin embargo, el mismo puede
ser mejorado, ajustando los parámetros de este modelo.

Nube de palabras
----------------

Con el fin de gráficar la información obtenida, la nube de palabras es
una buena opción en este caso, sólo se considerarán las palabras
positivas, en esta nube mientras más grande sea la palabra mayor fué su
uso.

    positive <- subset(tweetsSparse,tweetsSparse$sentiment==1)
    positive$sentiment <- NULL

    positivas <- as.data.frame(colSums(positive))
    positivas$words <- row.names(positivas)
    colnames(positivas) <- c("freq","word")

    wordcloud(positivas$word,positivas$freq,random.order = FALSE,colors = brewer.pal(8,"Dark2"),max.words = 300)

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : macbook could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : display could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : pour could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : never could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : amazon could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : money could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : dampen could not be fit on page. It will not be plotted.

    ## Warning in wordcloud(positivas$word, positivas$freq, random.order =
    ## FALSE, : someth could not be fit on page. It will not be plotted.

![](SVM_files/figure-markdown_strict/unnamed-chunk-14-1.png)
