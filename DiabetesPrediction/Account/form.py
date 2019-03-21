from django.forms import ModelForm

from predict.models import check
class InfoForm(ModelForm):
	class Meta:

		model=check
		fields=__all__  #[pregnancy,gl,bp,skin,insulen,bmi,dpf,age']  
