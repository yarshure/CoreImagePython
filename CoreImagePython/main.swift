/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Contains the main routine for setting up pycoreimage.
*/

import Foundation

func execute() {
    
    let script = Bundle.main.path(forResource: "pyci_demo", ofType: "py")
    let img = Bundle.main.path(forResource: "YourImagePath", ofType: "HEIC")
    let matte = Bundle.main.path(forResource: "YourFacePhotoPath", ofType: "HEIC")
    let dataset = URL(fileURLWithPath: img!).deletingLastPathComponent().path

    print(script!)
    print(img!)
    print(matte!)
    print(dataset)

    let pipe = Pipe()
    let file = pipe.fileHandleForReading;
    let task = Process()
    task.launchPath = "/usr/bin/python"
    task.arguments = [script, img, matte, dataset] as? [String]
    task.standardOutput = pipe
    task.launch()

    let data = file.readDataToEndOfFile()
    file.closeFile()
    
    let output = NSString(data: data, encoding: String.Encoding.utf8.rawValue)
    print(output!)
}

print("Hello, World!")
execute()
