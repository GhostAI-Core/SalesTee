import sys
import os
import inspect
import typing as t

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.ocap import OCapManager, Membrane

def audit_ambient_authority(obj: t.Any, disallowed_modules: t.List[str]) -> t.List[str]:
    """Inspects an object for leaked ambient authority (POLA audit)."""
    leaks = []
    # Check members for disallowed modules
    for name, member in inspect.getmembers(obj):
        if inspect.ismodule(member):
            module_name = member.__name__
            if any(module_name.startswith(d) for d in disallowed_modules):
                leaks.append(f"Leaked module via {name}: {module_name}")
    return leaks

def test_revocation_logic():
    print("=== OCap Security: Revocation Gate (Caretaker Pattern) ===\n")
    
    class ProtectedResource:
        def sensitive_data(self):
            return "TOP_SECRET_CORE_DATA"
            
    resource = ProtectedResource()
    
    # Simulate a revocation gate (Caretaker)
    class Caretaker:
        def __init__(self, target):
            self._target = target
            self.revoked = False
            
        def __getattr__(self, name):
            if self.revoked:
                raise PermissionError("Access Revoked: Caretaker gate closed.")
            return getattr(self._target, name)
            
    gate = Caretaker(resource)
    
    # Client only gets the gate
    print("[Test] Accessing resource through open gate...")
    print(f"Data: {gate.sensitive_data()}")
    
    print("[Test] Closing revocation gate...")
    gate.revoked = True
    
    try:
        print(f"Data: {gate.sensitive_data()}")
    except PermissionError as e:
        print(f"Verified: {e}")

def run_ocap_audits():
    print("\n=== OCap Security: POLA Ambient Authority Audit ===\n")
    
    from core.cell import AgenticCell
    from neural.orchestrator import orchestrator_singleton
    
    ocap_mgr = OCapManager()
    cell = AgenticCell("AuditTarget", ocap_mgr, orchestrator_singleton)
    
    # Audit for leaks of sensitive OS/System modules
    disallowed = ["os", "sys", "subprocess", "shutil"]
    leaks = audit_ambient_authority(cell, disallowed)
    
    if not leaks:
        print(f"[Audit] PASS: No ambient authority leaks found in AgenticCell.")
    else:
        for leak in leaks:
            print(f"[Audit] FAIL: {leak}")

if __name__ == "__main__":
    test_revocation_logic()
    run_ocap_audits()
