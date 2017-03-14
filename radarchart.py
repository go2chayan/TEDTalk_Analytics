#from ted_talk_experiments import function
from iplotter import ChartsJSPlotter

plotter = ChartsJSPlotter()

data = {
    "labels": ["anger", "disgust", "fear", "joy", "sadness", "analytical", "confident", "tentative", "openness", "conscientiousness", "extraversion", "agreeableness", "emotion"],
    "datasets": [
        {
            "label": "My First dataset",
            "fillColor": "rgba(125,195,160,0.2)",
            "strokeColor": "rgba(220,220,220,1)",
            "pointColor": "rgba(220,220,220,1)",
            "pointStrokeColor": "#fff",
            "pointHighlightFill": "#fff",
            "pointHighlightStroke": "rgba(220,220,220,1)",
            "data": [65, 59, 90, 81, 56, 55, 40, 10, 20, 30, 50, 20, 10]
        },
        {
            "label": "My Second dataset",
            "fillColor": "rgba(151,187,205,0.2)",
            "strokeColor": "rgba(151,187,205,1)",
            "pointColor": "rgba(151,187,205,1)",
            "pointStrokeColor": "#fff",
            "pointHighlightFill": "#fff",
            "pointHighlightStroke": "rgba(151,187,205,1)",
            "data": [28, 48, 40, 19, 96, 27, 100, 30, 50, 20, 10, 20, 10]
        }
    ]
}

plotter.plot_and_save(data, chart_type="Radar", w=500, h= 500, filename='radar', overwrite=True)
