from watson_developer_cloud import ToneAnalyzerV3
# NOTE: if your python system does not include watson_developer_cloud, please
# install it according to instructions given here: https://pypi.python.org/pypi/watson-developer-cloud
# Typically the following command should work (assumes pip):
# pip install --upgrade watson-developer-cloud

# Please get you own bluemix account (a free trial account will suffice) and update the
# username and password field
tone_analyzer = ToneAnalyzerV3(
   username='########-####-####-####-############',
   password='############',
   version='2016-05-19 ')
