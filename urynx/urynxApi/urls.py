from django.urls import path
from .import views
urlpatterns = [
    path('test/',views.test,name='test'),
    path('image/',views.ImamgeApi.as_view(),name='test'),

]
