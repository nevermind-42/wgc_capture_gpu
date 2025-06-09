#ifndef DXGI_ACCESS_H_
#define DXGI_ACCESS_H_
#include <Unknwn.h>

struct __declspec(uuid("a9b3d012-3df2-4ee3-b8d1-8695f457d3c1"))
IDirect3DDxgiInterfaceAccess : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE GetInterface(REFIID iid, void** p) = 0;
};

#endif // DXGI_ACCESS_H_ 