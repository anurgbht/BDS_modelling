# Class and method tutorial

class Employee:
    # these are changeable
    raise_amt = 1.04

    def __init__(self,first,last,pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'

    def fullname(self):
        return '{} {}'.format(self.first,self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)
        # if I had Employee.raise_amount then the changes would be for all the instances

    @classmethod
    # this is known as a decorator
    # class methods are alternative constructors
    def set_raise_amt(cls,amount):
        cls.raise_amt = amount

    @classmethod
    def from_string(cls, emp_str):
        first,last,pay = emp_str.split('-')
        return cls(first,last,pay)

class Developer(Employee):
    raise_amt = 1.1

    def __init__(self,first,last,pay,prog_lang):
        super().__init__(first,last,pay)
        self.prog_lang = prog_lang

class Manager(Employee):

    def __init__(self,first,last,pay,employees = None):
        super().__init__(first,last,pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees
    def add_emp(self,emp):
        self.employees.append(emp)

    def remove_emp(self,emp):
        self.employees.remove(emp)

    def print_employees(self):
        for emp in self.employees:
            print('-->',emp.fullname())


dev_1 = Developer('Corey','Schafer',5000,'Python')

mgr1 = Manager('Sur','dasrt',90000,[dev_1])

print(mgr1.email)

mgr1.print_employees()
