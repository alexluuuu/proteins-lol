"""
	scope_test.py
"""

def scope_testing():
	if True:
		a = 5
	else:
		a = 2
	print a

if __name__ == "__main__":
	scope_testing()
