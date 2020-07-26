class TestClass
{
	using My = TestClass;
	
	int a;
	My* pMy;
	int Foo(int a)
	{
		return a;
	}
};
