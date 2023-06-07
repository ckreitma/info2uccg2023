#https://pypi.org/project/sentiment-analysis-spanish/
#Instalación: pip install sentiment-analysis-spanish
from sentiment_analysis_spanish import sentiment_analysis

sentiment = sentiment_analysis.SentimentAnalysisSpanish()
print(sentiment.sentiment("me gusta la tómbola. La tómbola es la mejor del mundo"))
print(sentiment.sentiment("el puente de ñanduti es horrible"))
print(sentiment.sentiment("paraguay no ejecutó correctamente el plan COVID"))
print(sentiment.sentiment("el club guarani nunca será campeón"))
print(sentiment.sentiment("el club guarani no será eliminado"))
