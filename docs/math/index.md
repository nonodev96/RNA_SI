# \({L_{1}}\), \(L_{2}\), \(L_{p}\) and \(L_{\infty}\) in ML

En el mundo del Machine Learning, las normas \({L_{1}}\), \(L_{2}\), \(L_{p}\) y \(L_{\infty}\) juegan un papel fundamental en la regularización de modelos para controlar el sobreajuste, mejorar la generalización y optimizar el rendimiento predictivo. Estas normas son técnicas importantes que ayudan a mantener el equilibrio entre la complejidad del modelo y su capacidad para generalizar bien a datos no vistos. 

¿Qué son las Normas en el Contexto del Machine Learning?
En Machine Learning, las normas son funciones matemáticas que se utilizan para medir la magnitud de los coeficientes en un modelo. Al agregar términos de regularización basados en normas a la función de pérdida, los modelos de Machine Learning pueden simplificarse y evitar el sobreajuste al penalizar los coeficientes grandes.

## Norma \({L_{1}}\) (Norma de Valor Absoluto):

La norma \({L_{1}}\), también conocida como regularización Lasso (Least Absolute Shrinkage and Selection Operator), se define como la suma de los valores absolutos de los coeficientes del modelo. Al agregar la norma \({L_{1}}\) a la función de pérdida, se favorece la dispersión de los coeficientes, lo que puede llevar a la selección de características al forzar algunos coeficientes a cero. Esto hace que la norma \({L_{1}}\) sea útil para la selección automática de funciones y la creación de modelos más simples e interpretables.

## Norma \(L_{2}\) (Norma Euclidiana):

La norma \(L_{2}\), también conocida como regularización Ridge, se define como la raíz cuadrada de la suma de los cuadrados de los coeficientes del modelo. Al agregar la norma \(L_{2}\) a la función de pérdida, se penalizan los valores extremos de los coeficientes, lo que ayuda a suavizar el modelo y a reducir la sensibilidad a pequeñas variaciones en los datos. La regularización \(L_{2}\) es eficaz para prevenir el sobreajuste y mejorar la estabilidad del modelo.

## Norma \(L_{p}\) (Norma p):

La norma \(L_{p}\) generaliza las normas \({L_{1}}\) y \(L_{2}\), permitiendo ajustar la sensibilidad a la magnitud de los coeficientes mediante un parámetro p. Cuando p=1, obtenemos la norma \({L_{1}}\), y cuando p=2, obtenemos la norma \(L_{2}\). La norma \(L_{p}\) es útil cuando se necesita un control más flexible sobre la magnitud de los coeficientes en función de la situación específica del problema.

## Norma \(L_{\infty}\) (Norma Infinito):

La norma \(L_{\infty}\), también conocida como norma máxima, se define como el valor absoluto máximo de los coeficientes del modelo. Al utilizar la norma \(L_{\infty}\) en la regularización, se penalizan los coeficientes más grandes, lo que puede llevar a una mayor dispersión en los coeficientes y a una mayor resistencia al ruido en los datos.


- [Normas L1, L2, Lp y L∞ en Modelos de Machine Learning¡](https://www.linkedin.com/posts/naren-castellon-1541b8101_normas-l1-l2-lp-y-l-en-modelos-de-machine-activity-7241922894707011584-40eQ/?originalSubdomain=es)