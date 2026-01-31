
@app.get("/api/manifest/{project_id}")
async def get_manifest(project_id: str):
    """프로젝트 Manifest 조회"""
    manifest_path = Path(f"outputs/{project_id}/manifest.json")
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    try:
        content = manifest_path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {str(e)}")
