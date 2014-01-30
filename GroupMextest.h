// GroupMextest.h : main header file for the GroupMextest DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CGroupMextestApp
// See GroupMextest.cpp for the implementation of this class
//

class CGroupMextestApp : public CWinApp
{
public:
	CGroupMextestApp();

// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};
