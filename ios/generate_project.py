"""외부 프로젝트 생성기 없이 열 수 있는 Xcode 프로젝트를 재생성한다."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).parent

def uid(name):
    return hashlib.sha1(name.encode()).hexdigest()[:24].upper()

def quote(value):
    return json.dumps(str(value))

objects = []
def obj(name, fields):
    objects.append(f'{uid(name)} = {{ {fields} }};')
    return uid(name)

sources = sorted((ROOT / 'DocAgentApp').glob('*.swift'))
resources = ['Assets.xcassets', 'PrivacyInfo.xcprivacy', 'hand_landmarker.task']
file_ids = []
build_ids = []
resource_ids = []
for path in sources:
    ref = obj(path.name, f'isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = {quote(path.name)}; sourceTree = "<group>";')
    file_ids.append(ref)
    build_ids.append(obj('build-'+path.name, f'isa = PBXBuildFile; fileRef = {ref};'))
for name, kind in [('Info.plist','text.plist.xml'),('Assets.xcassets','folder.assetcatalog'),('PrivacyInfo.xcprivacy','text.xml'),('hand_landmarker.task','file')]:
    ref = obj(name, f'isa = PBXFileReference; lastKnownFileType = {kind}; path = {quote(name)}; sourceTree = "<group>";')
    file_ids.append(ref)
    if name in resources: resource_ids.append(obj('build-'+name, f'isa = PBXBuildFile; fileRef = {ref};'))
product = obj('product','isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = DocAgent.app; sourceTree = BUILT_PRODUCTS_DIR;')
app_group = obj('app-group',f'isa = PBXGroup; children = ({",".join(file_ids)}); path = DocAgentApp; sourceTree = "<group>";')
products = obj('products',f'isa = PBXGroup; children = ({product}); name = Products; sourceTree = "<group>";')
main = obj('main',f'isa = PBXGroup; children = ({app_group},{products}); sourceTree = "<group>";')
package = obj('core-package', 'isa = XCLocalSwiftPackageReference; relativePath = DocAgentCore;')
package_product = obj('core-product',f'isa = XCSwiftPackageProductDependency; package = {package}; productName = DocAgentCore;')
package_build = obj('core-build',f'isa = PBXBuildFile; productRef = {package_product};')
media = obj('mediapipe-package', 'isa = XCRemoteSwiftPackageReference; repositoryURL = "https://github.com/google-ai-edge/mediapipe.git"; requirement = { kind = revision; revision = 100f769b1532d5612ea5200958af7a536b804bb6; };')
media_product = obj('mediapipe-product',f'isa = XCSwiftPackageProductDependency; package = {media}; productName = MediaPipeTasksVision;')
media_build = obj('mediapipe-build',f'isa = PBXBuildFile; productRef = {media_product};')
frameworks = obj('frameworks',f'isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = ({package_build},{media_build}); runOnlyForDeploymentPostprocessing = 0;')
sources_phase = obj('sources',f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(build_ids)}); runOnlyForDeploymentPostprocessing = 0;')
resources_phase = obj('resources',f'isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(resource_ids)}); runOnlyForDeploymentPostprocessing = 0;')
for owner in ['project','target']:
    configs=[]
    for mode in ['Debug','Release']:
        settings={'SWIFT_VERSION':'5.0','IPHONEOS_DEPLOYMENT_TARGET':'17.0','SDKROOT':'iphoneos','CLANG_ENABLE_MODULES':'YES'}
        if owner=='target':
            settings.update(PRODUCT_BUNDLE_IDENTIFIER='org.sangsangground.DocAgent',PRODUCT_NAME='DocAgent',INFOPLIST_FILE='DocAgentApp/Info.plist',
                            CODE_SIGN_STYLE='Automatic',TARGETED_DEVICE_FAMILY='1',SUPPORTED_PLATFORMS='iphoneos iphonesimulator',
                            SUPPORTS_MACCATALYST='NO',SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD='NO',ASSETCATALOG_COMPILER_APPICON_NAME='AppIcon',
                            SWIFT_EMIT_LOC_STRINGS='YES',LD_RUNPATH_SEARCH_PATHS='$(inherited) @executable_path/Frameworks')
            # MediaPipe ships an arm64-only iOS simulator slice.
            settings['"EXCLUDED_ARCHS[sdk=iphonesimulator*]"']='x86_64'
        if mode=='Debug': settings.update(SWIFT_OPTIMIZATION_LEVEL='-Onone',SWIFT_ACTIVE_COMPILATION_CONDITIONS='DEBUG')
        configs.append(obj(owner+'-'+mode,'isa = XCBuildConfiguration; buildSettings = {'+' '.join(f'{k} = {quote(v)};' for k,v in settings.items())+'}; name = '+mode+';'))
    obj(owner+'-config',f'isa = XCConfigurationList; buildConfigurations = ({",".join(configs)}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target=obj('target',f'isa = PBXNativeTarget; buildConfigurationList = {uid("target-config")}; buildPhases = ({sources_phase},{frameworks},{resources_phase}); buildRules = (); dependencies = (); name = DocAgent; packageProductDependencies = ({package_product},{media_product}); productName = DocAgent; productReference = {product}; productType = "com.apple.product-type.application";')
project=obj('project',f'isa = PBXProject; attributes = {{ BuildIndependentTargetsInParallel = YES; LastUpgradeCheck = 1600; TargetAttributes = {{ {target} = {{ CreatedOnToolsVersion = 16.0; }}; }}; }}; buildConfigurationList = {uid("project-config")}; compatibilityVersion = "Xcode 14.0"; developmentRegion = ko; hasScannedForEncodings = 0; knownRegions = (ko,en,Base); mainGroup = {main}; packageReferences = ({package},{media}); productRefGroup = {products}; projectDirPath = ""; projectRoot = ""; targets = ({target});')
project_dir=ROOT/'DocAgent.xcodeproj';project_dir.mkdir(exist_ok=True)
(project_dir/'project.pbxproj').write_text('// !$*UTF8*$!\n{ archiveVersion = 1; classes = {}; objectVersion = 56; objects = {\n'+'\n'.join(objects)+f'\n}}; rootObject = {project}; }}\n')
scheme_dir=project_dir/'xcshareddata/xcschemes';scheme_dir.mkdir(parents=True,exist_ok=True)
(scheme_dir/'DocAgent.xcscheme').write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="1600" version="1.3">
<BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="DocAgent.app" BlueprintName="DocAgent" ReferencedContainer="container:DocAgent.xcodeproj"/></BuildActionEntry></BuildActionEntries></BuildAction>
<LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" allowLocationSimulation="NO"><BuildableProductRunnable runnableDebuggingMode="0"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="DocAgent.app" BlueprintName="DocAgent" ReferencedContainer="container:DocAgent.xcodeproj"/></BuildableProductRunnable></LaunchAction>
<ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"/>
<AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>''')
print(project_dir)
