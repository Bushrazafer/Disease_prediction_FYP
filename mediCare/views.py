# view for the home page
from django.http import HttpResponse
from django.shortcuts import render
def HomePage(request):
    data = {
        'title': 'Welcome to my website',
        'name': 'Ahmad Ali',
        'age': 6,
        'city': 'Multan',
        # list of hobbies
        'hobbies': ['Reading', 'Coding', 'Playing'],
        'is_active': True,
        # list of fruits
        'fruits': ['Apple', 'Banana', 'Orange', 'Grapes'],
        # dictionary of skills
        'skills': {
            'programming': 'Python',
            'web development': 'Django',
            'database': 'SQLite'
        },
        'number':[1,2,3,4,5,6,7,8,9,1]

    }
    return render(request, 'index.html', data)
def home(request):
    return HttpResponse("Hello, welcome to the home page!")
# slug dynamic URL
def home_slug(request, slug):
    return HttpResponse(f"This is the home page for : {slug}")
# view for the about page
def about(request):
    return HttpResponse("This is the about page.")

# view for the contact page
def contact(request):
    return HttpResponse("This is the contact page.")



# view for the contact page dynamically using an ID
def contactID(request, id):
    return HttpResponse(f"This is the contact page for ID: {id}")
# dynamic URL  for customer phone number
def customerPhone(request, phone_number):
    return HttpResponse(f"This is the customer page for phone number: {phone_number}")

